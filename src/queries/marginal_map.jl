export MMAP_LOG_HEADER, forward_bounds, edge_bounds, max_sum_lower_bound, associated_with_mult,
    mmap_solve, add_and_split, get_to_split, prune,
    gen_edges, brute_force_mmap, pc_condition, get_margs

using LogicCircuits
using StatsFuns: logaddexp
using DataStructures: DefaultDict, counter
using DataFrames
using Random

#####################
# Circuit Marginal Map
##################### 

"""
Info to be logged after every iteration of the solver
"""
MMAP_LOG_HEADER = ["iter", "prune_time", "split_time", "total_time",
    "prune_attempts", "num_edge_post_prune", "num_node_post_prune",
    "num_sum_post_prune", "num_max_post_prune", "ub_post_prune", "lb_post_prune",
    "split_var", "num_edge_post_split", "num_node_post_split",
    "num_sum_post_split", "num_max_post_split", "ub_post_split", "lb_post_split"]

#####################
# Marginal MAP bound computations
##################### 

"Computes the upper bound on MMAP probability for each edge. Returns a map from node or edge to its upper bound"
function edge_bounds(root::ProbCircuit, query_vars::BitSet, impl_lits=nothing)
    if isnothing(impl_lits)
        impl_lits = Dict()
        implied_literals(root, impl_lits)
    end
    mcache = forward_bounds(root, query_vars, impl_lits=impl_lits)
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


function forward_bounds(root::ProbCircuit, query_vars::BitSet; impl_lits=nothing, counters=nothing) 
    if isnothing(impl_lits)
        impl_lits = Dict()
        implied_literals(root, impl_lits)
    end
    if isnothing(counters)
        counters = Dict("max" => 0, "sum" => 0)
    end
    ret = forward_bounds_rec(root, query_vars, Dict{ProbCircuit, Float32}(), impl_lits, counters)
    @show counters
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

"""
Compute the lower bound on MMAP using a max-sum circuit
 
I.e. a forward bound that takes sum until encountering a max node.
If the circuit has a constrained vtree for the query variables, returns the exact MMAP.
"""
function max_sum_lower_bound(n::ProbCircuit, query_vars::BitSet)
    impl_lits = Dict()
    implied_literals(n, impl_lits)

    # forward pass 
    mcache = Dict{ProbCircuit,Tuple{Float32,Bool}}()
    max_sum_up(n, query_vars, impl_lits, mcache)

    # backward pass to get the max state
    state = zeros(Bool, num_variables(n))
    max_sum_down(n, mcache, state)

    # reduce to query variables
    reduced = transpose([i in query_vars ? state[i] : missing for i in 1:num_variables(n)])
    df = DataFrame(reduced)
    df, exp(MAR(n, df)[1])
end

function max_sum_up(root, query_vars, impl_lits, mcache)
    f_leaf(_) = (0.0f0, false)
    f_a(_, cs) = reduce((c1,c2) -> (c1[1]+c2[1], c1[2]||c2[2]), cs)
    f_o(n, cs) = begin
        has_max = any(last.(cs))
        if has_max || associated_with_mult(n, query_vars, impl_lits)
            val = reduce(max, Float32.(params(n)) .+ first.(cs))
            (val, true)
        else
            # propagate up as if every sum node adds up to 1
            (0.0f0, has_max)
        end
    end
    foldup_aggregate(root, f_leaf, f_leaf, f_a, f_o, Tuple{Float32,Bool}, mcache)
end

# function max_sum_rec(n::ProbCircuit, query_vars::BitSet, mcache::Dict{ProbCircuit, Tuple{Float32,Bool}}, impl_lits)
#     if isleaf(n)
#         mcache[n] = (0.0,false)
#     elseif isinner(n)
#         for c in n.children
#             if !haskey(mcache, c)
#                 max_sum_rec(c, query_vars, mcache, impl_lits)
#             end
#         end
#         if is⋀gate(n)
#             aggr_prod(c1,c2) = (c1[1]+c2[1], c1[2]||c2[2])
#             mcache[n] = mapreduce(c -> mcache[c], aggr_prod, n.children)
#         else
#             has_max = mapreduce(c->mcache[c][2], |, n.children)
#             if has_max || associated_with_mult(n, query_vars, impl_lits)
#                 val = mapreduce((c,p) -> mcache[c][1] + p, max, n.children, params(n))
#                 mcache[n] = (val, true)
#             else
#                 # propagate up as if every sum node adds up to 1
#                 mcache[n] = (0.0, has_max)
#             end
#         end
#     end
#     mcache
# end

function max_sum_down(n::ProbCircuit, mcache, state)
    if isleaf(n)
        state[variable(n)] = ispositive(n)
    elseif is⋀gate(n)
        for c in n.children
            max_sum_down(c, mcache, state)
        end
    else
        # If n is a max node, take the max branch.
        # Otherwise, n is a sum node that contains no query var or fixes some query vars.
        # In either case, any branch can be taken to retrieve the maximizing assignments to query vars.
        c_opt = n.children[1]
        m_opt = typemin(Float32)
        for (c,p) in zip(n.children, params(n))
            if mcache[c][1]+p >= m_opt
                c_opt = c
                m_opt = mcache[c][1]+p
            end
        end
        max_sum_down(c_opt, mcache, state)
    end
end

# "Check if a given sum node is associated with any query variables"

# function associated_with(n::ProbCircuit, query_vars::BitSet, impl_lits)
#     impl1 = impl_lits[n.children[1]]
#     impl2 = impl_lits[n.children[2]]
#     # First, we'll compute the set of variables that appear as a 
#     # positive implied literal on one side, and a negative implied literal on the other
#     neg_impl2 = BitSet(map(x -> -x, collect(impl2)))
#     decided_lits = intersect(impl1, neg_impl2)
#     decided_vars = BitSet(map(x -> abs(x), collect(decided_lits)))
#     # Now check if there's any overlap between these vars and the query
#     return !isempty(intersect(decided_vars, query_vars))
# end

# TODO: check using get_associated_vars?
"Check if a given sum node (with more than 2 children) is associated with any query variables"
function associated_with_mult(n::ProbCircuit, query_vars::BitSet, impl_lits)
    impl = [impl_lits[x] for x in n.children]
    neg_impl = [BitSet(map(x -> -x, collect(imp))) for imp in impl]
    # Checking all pairs
    for i in 1:num_children(n)
        for j in (i+1):num_children(n)
            decided_lits = intersect(impl[i], neg_impl[j])
            decided_vars = BitSet(map(x -> abs(x), collect(decided_lits)))
            if isdisjoint(decided_vars, query_vars)
                # If they don't differ in a max variable, not associated
                return false
            end
        end
    end
    true
end

#####################
# Edge Pruning
##################### 

function mmap_reduced_mpe_state(n::ProbCircuit, query_vars::BitSet)
    data_marg = DataFrame(repeat([missing], 1, num_variables(n)))
    mp, mappr = MAP(n, data_marg)
    # [i in query_vars ? mp[!, i][1] : missing for i in 1:num_variables(n)]
    l = collect(transpose([i in query_vars ? mp[!, i][1] : missing for i in 1:num_variables(n)]))
    df = DataFrame(l)
    df, exp(MAR(n, df)[1])
end

"Repeatedly prune a circuit using a given lower bound up to num_reps times"
function prune(n::PlainProbCircuit, query_vars::BitSet, lower_bound, num_reps=1)
    circ = n
    prev_size = num_edges(circ)
    counters = counter(Int)
    actual_reps = num_reps
    for i in 1:num_reps
        to_prune = find_to_prune(circ, query_vars, lower_bound, counters)
        circ = do_pruning(circ, Set(to_prune)) 

        if num_edges(circ) < prev_size
            prev_size = num_edges(circ)
        else  # early terminate if no more edges are getting pruned
            actual_reps = i
            break
        end
    end
    circ, counters, actual_reps
end

"Find edges to prune using a given lower bound (thresh) on the MMAP probability"
function find_to_prune(n::ProbCircuit, query_vars::BitSet, thresh, counters)
    impl_lits = Dict()
    implied_literals(n, impl_lits)

    rcache = edge_bounds(n, query_vars, impl_lits)
    bounds_list = collect(rcache)
    # Caution: if buffer is too small, edges may be pruned incorrectly. if it's too big, too little pruning may happen.
    to_prune = map(x -> x[1], filter(x -> isa(x[1], Tuple) && x[2] < thresh * (1 - 1e-4), bounds_list))

    query_lits = union(query_vars, BitSet(map(x->-x, collect(query_vars))))
    for (n,x) in to_prune
        decided_lits = setdiff(impl_lits[x], impl_lits[n])
        decided_query_lits = intersect(decided_lits, query_lits)
        merge!(counters, counter(decided_query_lits))
    end

    to_prune
end

"Perform the actual pruning. Note this will always return a plain logic circuit"
function do_pruning(n, to_prune)
    cache = Dict()
    foreach(x -> prune_fn(x, to_prune, cache), n)
    cache[n]
end

function prune_fn(n, to_prune, cache)
    if isinner(n)
        # Find children we are keeping
        inds = findall(x -> (n, x) ∉ to_prune, n.children)
        if isempty(inds)
            del_n = nothing     # no need to create a new node if all children are pruned
        elseif is⋁gate(n)
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

#####################
# Splitting on MMAP variables
##################### 

function get_associated_vars(n::ProbCircuit, query_vars::BitSet, impl_lits)
    if num_children(n) < 2
        return nothing
    end
    impl = [impl_lits[x] for x in n.children]
    neg_impl = [BitSet(map(x -> -x, collect(imp))) for imp in impl]
    vars = BitSet()
    # Checking all pairs
    for i in 1:num_children(n)
        for j in (i+1):num_children(n)
            decided_lits = intersect(impl[i], neg_impl[j])
            decided_vars = BitSet(map(x -> abs(x), collect(decided_lits)))
            decided_query_vars = intersect(decided_vars, query_vars)
            if isempty(decided_query_vars)
                # If they don't differ in a max variable, not associated
                return nothing
            else 
                union!(vars, decided_query_vars)
            end
        end
    end
    vars
end

"Add a unary parent node and then split on a variable"
function add_and_split(root, var)
    new_and = PlainMulNode([root])
    new_root = PlainSumNode([new_and])
    
    split_root = split(new_root, (new_root, new_and), Var(var), sanity_check=false, callback=keep_params, keep_unary=true)[1]
    split_root.log_probs .= 0   # each child has var as evidence and is left unnormalized
    split_root = remove_unary_gates(split_root)
    split_root
end


"Function to pass to condition so that it maintains parameters"
function keep_params(new_n, n, kept)
    new_n.log_probs = n.log_probs[kept]
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

function get_to_split(root, splittable, counters, heur)    
    max_prune(x,y) = begin
        counters[x] + counters[-x] >= counters[y] + counters[-y] ? x : y
    end

    min_diff(x,y) = begin
        diff_x = abs(counters[x] - counters[-x])
        diff_y = abs(counters[y] - counters[-y])
        if diff_x < diff_y
            x
        elseif diff_x == diff_y     # tie break by number of edges pruned
            max_prune(x,y)
        else
            y
        end 
    end

    pruned_vars = BitSet(map(x -> abs(x), collect(keys(counters))))
    pruned_splittable = intersect(splittable, pruned_vars)
    if heur == "maxDepth"
        # TODO: just get the max depth of each quer variable at the beginning of solver
        cache = Dict()
        impl_lits = Dict()
        implied_literals(root, impl_lits)
        max_depth_var(n, cache) = begin
            if isleaf(n)
                cache[n] = nothing
            else
                dec_nodes = filter(issomething, map(x -> cache[x], n.children))
                if isempty(dec_nodes)
                    vars = get_associated_vars(n, splittable, impl_lits)
                    cache[n] = isnothing(vars) ? nothing : (rand(vars),0)
                else
                    (var,depth) = reduce((c1,c2) -> c1[2] >= c2[2] ? c1 : c2, dec_nodes)
                    cache[n] = (var,depth+1)
                end
            end
        end
        foreach(n -> max_depth_var(n, cache), root)
        cache[root][1]
    elseif isempty(pruned_splittable) || heur == "rand"
        rand(splittable) 
    elseif heur == "minD"
        reduce(min_diff, collect(pruned_splittable))
    else # maxP
        reduce(max_prune, collect(pruned_splittable))
    end
end

#####################
# Iterative MMAP solver
##################### 

"Update upper and lower bounds, and add log info"
function update_and_log(cur_root, root, quer, lb, results, iter, post_prune; time=missing, prune_attempts=missing, split_var=missing)
    # Recompute upper and lower bounds
    counters = Dict("max" => 0, "sum" => 0)
    ub = forward_bounds(cur_root, quer, counters=counters)[cur_root]    # TODO: could probably reuse impl_lits without recomputing
    mstate, mp = max_sum_lower_bound(cur_root, quer)
    new_lb = max(lb, MAR(root, mstate)[1]) # use the original circuit (before pruning) for lower bounds

    if post_prune
        results["iter"][iter] = iter
        results["prune_time"][iter] = time
        results["prune_attempts"][iter] = prune_attempts
        postfix = "_post_prune"
    else # post_split
        results["split_time"][iter] = time
        results["total_time"][iter] = (iter==1 ? 0 : results["total_time"][iter-1]) + results["prune_time"][iter] + time
        results["split_var"][iter] = split_var
        postfix = "_post_split"
    end

    results[string("num_edge",postfix)][iter] = num_edges(cur_root)
    results[string("num_node",postfix)][iter] = num_nodes(cur_root)
    results[string("num_sum",postfix)][iter] = counters["sum"]
    results[string("num_max",postfix)][iter] = counters["max"]
    results[string("ub",postfix)][iter] = ub
    results[string("lb",postfix)][iter] = lb

    ub, new_lb
end

"Compute marginal MAP by iteratively pruning and splitting"
function mmap_solve(root, quer; num_iter=length(quer), prune_attempts=10, log_per_iter=noop, heur="maxP")
    # initialize log
    results = Dict()
    for x in MMAP_LOG_HEADER
        results[x] = Vector{Union{Any,Missing}}(missing,num_iter)
    end

    splittable = copy(quer)
    cur_root = root
    ub = forward_bounds(cur_root, quer)[cur_root]
    mstate, mp = max_sum_lower_bound(cur_root, quer)
    # mstate, mp = mmap_reduced_mpe_state(cur_root, quer)
    lb = log(mp)
    @show ub, lb
    for i in 1:num_iter
        try
            if isempty(splittable) || ub < lb + 1e-10
                break
            end

            # Prune -- could improve the upper bound (TODO: maybe also the lower bound?)
            println("* Starting with $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
            tic = time_ns()
            cur_root, counters, actual_reps = prune(cur_root, quer, exp(lb), prune_attempts)
            toc = time_ns()
            println("* Pruning gives $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
            @show update_and_log(cur_root,root,quer,lb,results,i,true, prune_attempts=actual_reps, time=(toc-tic)/1.0e9)

            # Split root (move a quer variable up) -- could improve both the upper and lower bounds
            tic = time_ns()
            to_split = get_to_split(cur_root, splittable, counters, heur)
            # to_split = get_to_split(root, splittable, counters, heur)
            cur_root = add_and_split(cur_root, to_split)
            toc = time_ns()
            println("* Splitting on $(to_split) gives $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
            delete!(splittable, to_split)
            @show ub, lb = update_and_log(cur_root,root,quer,lb,results,i,false, split_var=to_split, time=(toc-tic)/1.0e9)

            log_per_iter(results)
            # TODO: also save the circuit at the end of each iteration for easy retrieval?    
        catch e
            println(sprint(showerror, e, backtrace()))
            break
        end
    end
    cur_root
end

#####################
# Useful functions for testing
##################### 

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

# "Do a bottom up pass to renormalize parameters, correctly propagating
# This follows algorithm 1 from 'On Theoretical Properties of Sum-Product Networks'"
# function bottomup_renorm_params(root)
#     f_con(_) = 0.0
#     f_lit(_) = 0.0
#     f_a(_, cn) = sum(cn)
#     f_o(n, cn) = begin
#         n.log_probs = n.log_probs .+ cn
#         alpha = reduce(logaddexp, n.log_probs)
#         n.log_probs = n.log_probs .- alpha
#         alpha
#     end
#     foldup_aggregate(root, f_con, f_lit, f_a, f_o, Float64)
#     root
# end

pc_condition(root::PlainProbCircuit, var1, var2, var3...) = pc_condition(pc_condition(root, var1), var2, var3...)

function pc_condition(root::PlainProbCircuit, var)
    conjoin(root, var2lit(var), callback=keep_params, keep_unary=true)
    # condition(root, var2lit(var))
end

function get_margs(root, num_vars, vars, cond, norm=false)
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