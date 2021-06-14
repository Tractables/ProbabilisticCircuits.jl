export forward_bounds, edge_bounds, prune_mpe, 
    do_pruning, rep_mpe_pruning, gen_edges, brute_force_mmap, associated_with_mult,
    add_and_split, remove_unary_gates, pc_condition, get_margs, normalize_params, bottomup_renorm_params,
    map_mpe_random

using LogicCircuits
using StatsFuns: logaddexp
using DataStructures: DefaultDict
using DataFrames
using Random

#####################
# Circuit Marginal Map
##################### 

# Everything to do with search based marginal map computation

# 
function edge_bounds(root::ProbCircuit, query_vars::BitSet)
    impl_lits = Dict()
    implied_literals(root, impl_lits)
    mcache = forward_bounds(root, query_vars)
    tcache = DefaultDict{ProbCircuit, Float64}(0.0)
    tcache[root] = 1.0
    rcache = DefaultDict{Union{ProbCircuit, Tuple{ProbCircuit, ProbCircuit}}, Float64}(0.0)
    rcache[root] = exp(mcache[root])
    foreach_down(x -> edge_bounds_fn(x, query_vars, impl_lits, mcache, tcache, rcache), root)
    rcache
end


function edge_bounds_fn(root::ProbCircuit, query_vars::BitSet, impl_lits, mcache,
    tcache::DefaultDict{ProbCircuit, Float64, Float64},
    rcache::DefaultDict{Union{ProbCircuit, Tuple{ProbCircuit, ProbCircuit}}, Float64, Float64})
    if isleaf(root)
        return 
    end
    if tcache[root] == 0.0
        return
    end
    if is⋁gate(root)
        for (c, param) in zip(root.children, params(root))
            if num_children(root) >= 2 && associated_with_mult(root, query_vars, impl_lits)
                if(abs(exp(param) * exp(mcache[c]) - exp(mcache[root])) > 1e-7)
                    rcache[(root, c)] = rcache[root] + tcache[root] * (exp(param) * exp(mcache[c]) - exp(mcache[root]))
                else
                    rcache[(root, c)] = rcache[root]
                end
            else
                rcache[(root, c)] = rcache[root]
            end
            if mcache[c] > -Inf
                if tcache[c] == 0 
                    tcache[c] = exp(param) * tcache[root]
                else
                    tcache[c] = min(tcache[c], exp(param) * tcache[root])
                end
            end
            rcache[c] = max(rcache[c], rcache[(root, c)])
        end
    else
        for c in root.children
            rcache[(root, c)] = rcache[root]
            if tcache[c] == 0
                tcache[c] = tcache[root]
            else
                tcache[c] = min(tcache[c], tcache[root])
            end
            rcache[c] = max(rcache[c], rcache[(root, c)])
        end
    end
end


function forward_bounds(root::ProbCircuit, query_vars::BitSet) 
    impl_lits = Dict()
    implied_literals(root, impl_lits)
    counters = Dict("max" => 0, "sum" => 0)
    ret = forward_bounds_rec(root, query_vars, Dict{ProbCircuit, Float32}(), impl_lits, counters)
    # @show counters
    ret
end


function forward_bounds_rec(root::ProbCircuit, query_vars::BitSet, mcache::Dict{ProbCircuit, Float32}, impl_lits, counters)
    if isleaf(root)
        mcache[root] = 0.0
    elseif isinner(root)
        for c in root.children
            if !haskey(mcache, c)
                forward_bounds_rec(c, query_vars, mcache, impl_lits, counters)
            end
        end
        if is⋀gate(root) 
            mcache[root] = mapreduce(c -> mcache[c], +, root.children)
        else
            # @assert(num_children(root) <= 2)
            # If we have just the one child, just incorporate the parameter
            if num_children(root) == 1
                mcache[root] = mcache[root.children[1]] + params(root)[1]
            else
                # If we have 2 children, check if associated:
                if associated_with_mult(root, query_vars, impl_lits)
                    # Max node
                    counters["max"] = counters["max"] + 1 
                    # If it is, we're taking a max
                    mcache[root] = mapreduce((c,p) -> mcache[c] + p, max, root.children, params(root))
                else
                    # If it isn't, we're taking a sum
                    counters["sum"] = counters["sum"] + 1
                    # @show params(root)
                    mcache[root] = mapreduce((c,p) -> mcache[c] + p, logaddexp, root.children, params(root))
                    # mcache[root] = logsumexp()mapreduce((c,p) -> mcache[c] + p, logaddexp, root.children, params(root))
                    # @show mcache[root]
                end
            end
        end
    end
    mcache
end

"Check if a given sum node is associated with any query variables"

function associated_with(n::ProbCircuit, query_vars::BitSet, impl_lits)
    impl1 = impl_lits[n.children[1]]
    impl2 = impl_lits[n.children[2]]
    # First, we'll compute the set of variables that appear as a 
    # positive implied literal on one side, and a negative implied literal on the other
    neg_impl2 = BitSet(map(x -> -x, collect(impl2)))
    decided_lits = intersect(impl1, neg_impl2)
    decided_vars = BitSet(map(x -> abs(x), collect(decided_lits)))
    # Now check if there's any overlap between these vars and the query
    return !isempty(intersect(decided_vars, query_vars))
end

"Check if a given sum node (with more than 2 children) is associated with any query variables"

function associated_with_mult(n::ProbCircuit, query_vars::BitSet, impl_lits)
    impl = [impl_lits[x] for x in n.children]
    neg_impl = [BitSet(map(x -> -x, collect(imp))) for imp in impl]
    # Checking all pairs
    for i in 1:num_children(n)
        for j in (i+1):num_children(n)
            decided_lits = intersect(impl[i], neg_impl[j])
            decided_vars = BitSet(map(x -> abs(x), collect(decided_lits)))
            if isempty(intersect(decided_vars, query_vars))
                # If they don't differ in a max variable, not associated
                return false
            end
        end
    end
    true
end

"Find edges to prune using MPE state reduced to query variables"

function prune_mpe(n::ProbCircuit, query_vars::BitSet)
    # First get the mpe state reduced to query variables, then the threshold
    # data_marg = DataFrame(repeat([missing], 1, num_variables(n)))
    # mp, mappr = MAP(n, data_marg)
    # [i in query_vars ? mp[!, i][1] : missing for i in 1:num_variables(n)]
    # l = collect(transpose([i in query_vars ? mp[!, i][1] : missing for i in 1:num_variables(n)]))
    # df = DataFrame(l)
    @show thresh = mmap_reduced_mpe_state(n, query_vars)

    rcache = edge_bounds(n, query_vars)
    bounds_list = collect(rcache)
    map(x -> x[1], filter(x -> isa(x[1], Tuple) && x[2] < thresh - 1e-9, bounds_list))
    # filter(x -> isa(x[1], Tuple) && x[2] < thresh, bounds_list)
end

function mmap_reduced_mpe_state(n::ProbCircuit, query_vars::BitSet)
    data_marg = DataFrame(repeat([missing], 1, num_variables(n)))
    mp, mappr = MAP(n, data_marg)
    [i in query_vars ? mp[!, i][1] : missing for i in 1:num_variables(n)]
    l = collect(transpose([i in query_vars ? mp[!, i][1] : missing for i in 1:num_variables(n)]))
    df = DataFrame(l)
    exp(MAR(n, df)[1])
end

function rep_mpe_pruning(n::PlainProbCircuit, query_vars::BitSet, num_reps)
    circ = n
    for i in 1:num_reps
        res = prune_mpe(circ, query_vars)
        to_prune = Set(res)
        circ = do_pruning(circ, to_prune)
    end
    circ
end

"Preform the actual pruning. Note this will always return a plain logic circuit"

function do_pruning(n, to_prune)
    cache = Dict()
    foreach(x -> prune_fn(x, to_prune, cache), n)
    cache[n]
end

function prune_fn(n, to_prune, cache)
    if isinner(n)
        # Find children we are keeping
        inds = findall(x -> (n, x) ∉ to_prune, n.children)
        if is⋁gate(n)
            del_n = PlainSumNode(map(x -> cache[x], n.children[inds]))
            del_n.log_probs = n.log_probs[inds]
        else
            del_n = PlainMulNode(map(x -> cache[x], n.children[inds]))
        end
        cache[n] = del_n
    else
        # Leaf nodes just use themselves, fine to reuse in new circuit
        cache[n] = n
    end
end

"Generate a list of (parent, child) pairs representing edges"

function gen_edges(root)
    edges = []
    foreach(root) do n
        if isinner(n)
            edges = vcat(edges, map(x -> (n, x), children(n)))
        end
    end
    edges
end

"Manually compute the marginal map probability by brute force"

function brute_force_mmap(root, query_vars)
    quer_data_all = generate_data_all(length(query_vars))
    qda = convert(Matrix, quer_data_all)
    result = DataFrame(missings(Bool, 1 << length(query_vars), num_variables(root)))
    result[:, collect(query_vars)] = qda
    @show result
    @show MAR(root, result)
    reduce(max, MAR(root, result))
end

"Add a unary parent node and then split on a variable"

function add_and_split(root, var)
    new_and = PlainMulNode([root])
    new_root = PlainSumNode([new_and])
    df = DataFrame(missings(Bool, 1, num_variables(root)))
    df[1, var] = true
    # @show df
    # @show pos_mar = MAR(root, df)
    
    split_root = split(new_root, (new_root, new_and), Var(var), callback=keep_params, keep_unary=true)[1]
        # split_root.log_probs = [pos_mar[1], log(1-exp(pos_mar[1]))]
    data_marg = DataFrame(repeat([missing], 1, num_variables(root)))
    mp, mappr = MAP(root, data_marg)
    @show mappr
    split_root = bottomup_renorm_params(split_root)
    mp, mappr = MAP(root, data_marg)
    @show mappr

    split_root = remove_unary_gates(split_root)
    mp, mappr = MAP(root, data_marg)
    @show mappr
    split_root
end

"Function to pass to condition so that it maintains and normalizes parameters"
function fix_params(new_n, n, kept)
    total_prob = reduce(logaddexp, n.log_probs[kept])
    new_n.log_probs = map(x -> x - total_prob, n.log_probs[kept])
end

"Function to pass to condition so that it maintains parameters"
function keep_params(new_n, n, kept)
    new_n.log_probs = n.log_probs[kept]
end

"Normalize params"
function normalize_params(root)
    f_con(n) = n
    f_lit(n) = n
    f_a(n, cn) = multiply([cn...])
    f_o(n, cn) = begin
        ret = summate([cn...])
        total_prob = reduce(logaddexp, n.log_probs)
        ret.log_probs = map(x -> x - total_prob, n.log_probs)
        ret
    end
    foldup_aggregate(root, f_con, f_lit, f_a, f_o, Node)
end

"Do a bottom up pass to renormalize parameters, correctly propagating
This follows algorithm 1 from 'On Theoretical Properties of Sum-Product Networks'"
function bottomup_renorm_params(root)
    f_con(_) = 0.0
    f_lit(_) = 0.0
    f_a(_, cn) = sum(cn)
    f_o(n, cn) = begin
        n.log_probs = n.log_probs .+ cn
        alpha = reduce(logaddexp, n.log_probs)
        n.log_probs = n.log_probs .- alpha
        alpha
    end
    foldup_aggregate(root, f_con, f_lit, f_a, f_o, Float64)
    root
end

"Return an equivalent circuit without and nodes with single children"

function remove_unary_gates(root::PlainProbCircuit)
    f_con(n) = n
    f_lit(n) = n
    f_a(n, cn) = begin 
        if length(cn) == 1 
            cn[1]
        else
            multiply([cn...], reuse=n)
        end
    end
    f_o(n, cn) = begin 
        ret = summate([cn...], reuse=n)
        ret.log_probs = n.log_probs
        ret
    end
    foldup_aggregate(root, f_con, f_lit, f_a, f_o, Node)
end

pc_condition(root::PlainProbCircuit, var1, var2, var3...) = pc_condition(pc_condition(root, var1), var2, var3...)

function pc_condition(root::PlainProbCircuit, var)
    conjoin(root, var2lit(var), callback=keep_params, keep_unary=true)
    # condition(root, var2lit(var))
end

function get_margs(root, num_vars, vars, cond, norm=true)
    quer_data_all = generate_data_all(length(vars))
    qda = convert(Matrix, quer_data_all)
    result = DataFrame(missings(Bool, 1 << length(vars), num_vars))
    result[:, vars] = qda
    if length(cond) > 0
        result[:, cond] .= true
    end
    result
    mars = MAR(root, result)
    if norm
        mars = mars .- reduce(logaddexp, mars)
    end
    mars
end

function map_mpe_random(root, quer)
    splittable = copy(collect(quer))
    while true
        println("Forward bound at root: $(forward_bounds(root, quer)[root])")
        println("MPE: $(log(mmap_reduced_mpe_state(root, quer)))")
        data_marg = DataFrame(repeat([missing], 1, num_variables(root)))
        mp, mappr = MAP(root, data_marg)
        @show mappr

        split_ind = rand(1:length(splittable))
        to_split = splittable[split_ind]
        println("Starting with $(num_edges(root)) edges and $(num_nodes(root)) nodes.")
        root = rep_mpe_pruning(root, quer, 1)
        println("Pruning gives $(num_edges(root)) edges and $(num_nodes(root)) nodes.")
        println("Forward bound at root: $(forward_bounds(root, quer)[root])")
        println("MPE: $(log(mmap_reduced_mpe_state(root, quer)))")
        data_marg = DataFrame(repeat([missing], 1, num_variables(root)))
        mp, mappr = MAP(root, data_marg)
        @show mappr

        root = add_and_split(root, to_split)
        println("Splitting on $(to_split) gives $(num_edges(root)) edges and $(num_nodes(root)) nodes.")
        deleteat!(splittable, split_ind)
    end
end