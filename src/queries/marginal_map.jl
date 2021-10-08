export MMAP_LOG_HEADER, forward_bounds, edge_bounds, max_sum_lower_bound, associated_with_mult,
    mmap_solve, add_and_split, get_to_split, prune,
    mmap_reduced_mpe_state, gen_edges, brute_force_mmap, pc_condition, get_margs,
    marginalize_out

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

NodeCacheDict = DefaultDict{ProbCircuit, Float64}
NodeEdgeCacheDict = DefaultDict{Union{ProbCircuit, Tuple{ProbCircuit, ProbCircuit}}, Float64}
LitCacheDict = Dict{ProbCircuit, Union{BitSet, Nothing}}

struct MMAPCache
    ub::Dict{ProbCircuit, Float32}  # cache for upper bound forward pass
    lb::Dict{ProbCircuit,Tuple{Float32,Bool}}  # cache for lower bound forward pass (max-sum)
    impl_lits::LitCacheDict         # implied literals
end

MMAPCache() = MMAPCache(Dict{ProbCircuit, Float32}(), Dict{ProbCircuit,Tuple{Float32,Bool}}(), LitCacheDict())

#####################
# Marginal MAP bound computations
##################### 

"Computes the upper bound on MMAP probability for each edge. Returns a map from node or edge to its upper bound"
function edge_bounds(root::ProbCircuit, query_vars::BitSet, cache::MMAPCache)
    implied_literals(root, cache.impl_lits)
    forward_bounds(root, query_vars, cache)

    # tcache = NodeCacheDict(0.0)
    # tcache[root] = 1.0
    # rcache = NodeEdgeCacheDict(0.0)
    # rcache[root] = exp(cache.ub[root])

    tcache = NodeCacheDict(-Inf)
    tcache[root] = 0.0
    rcache = NodeEdgeCacheDict(-Inf)
    rcache[root] = cache.ub[root]

    foreach_down(x -> edge_bounds_fn(x, query_vars, cache, tcache, rcache), root)
    # @assert all(collect(values(rcache)) .>= 0.0)
    rcache
end

function edge_bounds_fn(root::ProbCircuit, query_vars::BitSet, cache::MMAPCache, tcache::NodeCacheDict, rcache::NodeEdgeCacheDict)
    # if isleaf(root) || tcache[root] == 0.0
    if isleaf(root) || tcache[root] == -Inf
        return
    end
    if is⋁gate(root)
        for (c, param) in zip(root.children, params(root))
            if (associated_with_mult(root, query_vars, cache.impl_lits)
                && cache.ub[root] - (param + cache.ub[c]) > 1e-5)
                # rcache[(root, c)] = rcache[root] + tcache[root] * (exp(param) * exp(cache.ub[c]) - exp(cache.ub[root]))
                rcache[(root, c)] = logsubexp(rcache[root], tcache[root] + logsubexp(param + cache.ub[c], cache.ub[root]))
                # rcache[(root, c)] = logsubexp(logaddexp(rcache[root], tcache[root] + param + cache.ub[c]), tcache[root] + cache.ub[root])
            else
                rcache[(root, c)] = rcache[root]
            end
            if cache.ub[c] > -Inf
                # edge_pr = exp(param) * tcache[root]
                # tcache[c] = tcache[c] == 0 ? edge_pr : min(tcache[c], edge_pr)
                edge_pr = param + tcache[root]
                tcache[c] = tcache[c] > -Inf ? min(tcache[c], edge_pr) : edge_pr
            end
            rcache[c] = max(rcache[c], rcache[(root, c)])
        end
    else
        for c in root.children
            rcache[(root, c)] = rcache[root]
            rcache[c] = max(rcache[c], rcache[(root, c)])
            # tcache[c] = tcache[c] == 0 ? tcache[root] : min(tcache[c], tcache[root])
            tcache[c] = tcache[c] > -Inf ? min(tcache[c], tcache[root]) : tcache[root]
        end
    end
end

# TODO: perhaps a flag to skip recomputing impl_lits
function forward_bounds(root::ProbCircuit, query_vars::BitSet, cache::Union{MMAPCache, Nothing}=nothing) 
    if isnothing(cache)
        cache = MMAPCache()
    end

    implied_literals(root, cache.impl_lits)

    f_leaf(_) = 0.0f0
    f_a(_, cs) = sum(cs)
    f_o(n, cs) = begin
        if associated_with_mult(n, query_vars, cache.impl_lits)
            maximum(Float32.(params(n)) .+ cs)
        else
            reduce(logaddexp, Float32.(params(n)) .+ cs)
        end
    end
    foldup_aggregate(root, f_leaf, f_leaf, f_a, f_o, Float32, cache.ub)
end

function forward_bounds(root::ProbCircuit, query_vars::BitSet, data::DataFrame, impl_lits=nothing) 
    if isnothing(impl_lits)
        impl_lits = LitCacheDict()
    end
    implied_literals(root, impl_lits)
    
    @assert num_features(data) == maximum(variables(root))
    f_con(n) = zeros(Float32, nrow(data))   # Note: no constant node for ProbCircuit anyway
    f_lit(n) = begin
        assignments = ispositive(n) ? data[:,variable(n)] : .!data[:,variable(n)]
        log.(Float32.(coalesce.(assignments, true)))
    end
    f_a(_, cs) = sum(cs)
    f_o(n, cs) = begin
        evals = map((p,cv) -> p .+ cv, Float32.(params(n)), cs)
        if associated_with_mult(n, query_vars, impl_lits)
            reduce((x,y) -> max.(x,y), evals)
        else
            reduce((x,y) -> logaddexp.(x,y), evals)
        end
    end
    foldup_aggregate(root, f_con, f_lit, f_a, f_o, Array{Float32})
end

"""
Compute the lower bound on MMAP using a max-sum circuit
 
I.e. a forward bound that takes sum until encountering a max node.
If the circuit has a constrained vtree for the query variables, returns the exact MMAP.
"""
function max_sum_lower_bound(root::ProbCircuit, query_vars::BitSet, cache::Union{MMAPCache, Nothing}=nothing)
    if isnothing(cache)
        cache = MMAPCache()
    end

    implied_literals(root, cache.impl_lits)

    # forward pass 
    f_leaf(_) = (0.0f0, false)
    f_a(_, cs) = reduce((c1,c2) -> (c1[1]+c2[1], c1[2]||c2[2]), cs)
    f_o(n, cs) = begin
        has_max = any(last.(cs))    # whether n has a max node as a descendant
        if has_max || associated_with_mult(n, query_vars, cache.impl_lits)
            val = reduce(max, Float32.(params(n)) .+ first.(cs))
            (val, true)
        else
            val = reduce(logaddexp, Float32.(params(n)) .+ first.(cs))
            (val, false)
        end
    end
    foldup_aggregate(root, f_leaf, f_leaf, f_a, f_o, Tuple{Float32,Bool}, cache.lb)

    # backward pass to get the max state
    # state = Array{Union{Nothing,Bool}}(nothing, num_variables(root))
    state = Array{Union{Nothing,Bool}}(nothing, maximum(variables(root)))
    max_sum_down(root, cache.lb, state)

    # reduce to query variables
    @assert all(issomething.(state[collect(query_vars)]))
    # reduced = transpose([i in query_vars ? state[i] : missing for i in 1:num_variables(root)])
    reduced = transpose([i in query_vars ? state[i] : missing for i in 1:maximum(variables(root))])
    df = DataFrame(reduced, :auto)
    df, exp(MAR(root, df)[1])
end

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
    if num_children(n) < 2
        return false
    end

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
function prune(n::PlainProbCircuit, query_vars::BitSet, cache::MMAPCache, lower_bound, num_reps=1)
    circ = n
    prev_size = num_edges(circ)
    counters = counter(Int)
    actual_reps = num_reps
    for i in 1:num_reps
        to_prune = find_to_prune(circ, query_vars, cache, lower_bound, counters)
        circ = do_pruning(circ, Set(to_prune), cache) 

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
function find_to_prune(n::ProbCircuit, query_vars::BitSet, cache::MMAPCache, thresh, counters)
    implied_literals(n, cache.impl_lits)

    rcache = edge_bounds(n, query_vars, cache)
    # Caution: if buffer is too small, edges may be pruned incorrectly. if it's too big, too little pruning may happen.
    # to_prune = map(x -> x[1], filter(x -> isa(x[1], Tuple) && x[2] < thresh * (1 - 1e-4), collect(rcache)))
    to_prune = map(x -> x[1], filter(x -> isa(x[1], Tuple) && x[2] < thresh - 1e-5, collect(rcache)))

    query_lits = union(query_vars, BitSet(map(x->-x, collect(query_vars))))
    for (n,x) in to_prune
        decided_lits = setdiff(cache.impl_lits[x], cache.impl_lits[n])
        decided_query_lits = intersect(decided_lits, query_lits)
        merge!(counters, counter(decided_query_lits))
    end

    to_prune
end

"Perform the actual pruning. Note this will always return a plain logic circuit"
function do_pruning(n, to_prune, cache)
    new_nodes = Dict()
    foreach(x -> prune_fn(x, to_prune, new_nodes), n)

    # Only keep in cache nodes that are not pruned
    filter!(p -> issomething(get(new_nodes, p.first, nothing)), cache.ub)
    filter!(cache.lb) do p
        if p isa Tuple
            (issomething(get(new_nodes, p.first.first, nothing))
             && issomething(get(new_nodes, p.first.second, nothing)))
        else
            issomething(get(new_nodes, p.first, nothing))
        end
    end
    filter!(p -> issomething(get(new_nodes, p.first, nothing)), cache.impl_lits)
    new_nodes[n]
end

function prune_fn(n, to_prune, cache)
    if isinner(n)
        # Find children we are keeping
        inds = findall(x -> (n, x) ∉ to_prune, n.children)
        new_children = map(x -> cache[x], n.children[inds])
        if isempty(inds)
            cache[n] = nothing  # no need to create a new node if all children are pruned
        elseif new_children == n.children
            cache[n] = n        # reuse node if no edges below are pruned
        elseif is⋁gate(n)
            del_n = PlainSumNode(map(x -> cache[x], n.children[inds]))
            del_n.log_probs = n.log_probs[inds]
            cache[n] = del_n
        else
            cache[n] = PlainMulNode(map(x -> cache[x], n.children[inds]))
        end
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

function get_to_split(root, splittable, counters, heur, lb)    
    if heur == "avgUB" || heur == "minUB" || heur == "maxUB" || heur == "UB"
        vars = collect(splittable)
        datamat = Array{Union{Missing, Bool}}(missing, 2*length(vars), maximum(variables(root)))
        for i in 1:length(vars)
            datamat[2*i-1, vars[i]] = true
            datamat[2*i, vars[i]] = false
        end
        bounds = forward_bounds(root, splittable, DataFrame(datamat, :auto))
        if heur == "UB"
            prunable = filter(i -> min(bounds[2*i-1], bounds[2*i]) < lb * (1 - 1e-4), 1:length(vars))
            if isempty(prunable)
                idx = reduce((i,j) -> logaddexp(bounds[2*i-1], bounds[2*i]) < logaddexp(bounds[2*j-1], bounds[2*j]) ? i : j, 1:length(vars))
            else
                idx = reduce((i,j) -> max(bounds[2*i-1], bounds[2*i]) < max(bounds[2*j-1], bounds[2*j]) ? i : j, prunable)
            end
        else
            op = Dict("avgUB" => logaddexp, "minUB" => min, "maxUB" => max)
            idx = reduce((i,j) -> op[heur](bounds[2*i-1], bounds[2*i]) < op[heur](bounds[2*j-1], bounds[2*j]) ? i : j, 1:length(vars))
        end
        vars[idx]
    
    elseif heur == "maxDepth"
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

    else
        pruned_vars = BitSet(map(x -> abs(x), collect(keys(counters))))
        pruned_splittable = intersect(splittable, pruned_vars)
        
        max_prune(x,y) = counters[x] + counters[-x] >= counters[y] + counters[-y] ? x : y
    
        min_diff(x,y) = begin
            min_x = min(counters[x], counters[-x])
            min_y = min(counters[y], counters[-y])
            if min_x > min_y
                x
            elseif min_x == min_y
                max_prune(x,y)
            else
                y
            end
        end
        
        if isempty(pruned_splittable) || heur == "rand"
            rand(splittable) 
        elseif heur == "minD"
            reduce(min_diff, collect(pruned_splittable))
        else # maxP
            reduce(max_prune, collect(pruned_splittable))
        end
    end
end

#####################
# Iterative MMAP solver
##################### 

"""
Marginalize out variables. Requires at least one variable that is not marginalized.

Requires and preserves smoothness and decomposability. This may break determinism. 
"""
function marginalize_out(root, to_marginalize)
    # TOOD: take into account evidence
    # NOTE: without evidence, marg should always be 0.0
    f_leaf(n) = variable(n) ∈ to_marginalize ? (0.0f0, nothing) : (0.0f0, n)
    f_a(n, cn) = begin
        children = filter(issomething, last.(cn))
        if isempty(children)
            new_n = nothing
        elseif length(children) == 1
            new_n = children[1]
        else
            new_n = multiply(children, reuse=n)
        end
        marg = sum(first.(cn))
        (marg, new_n)
    end
    f_o(n, cn) = begin
        # By smoothness, either all children are marginalized (ie nothing) or none are.
        if all(isnothing.(last.(cn)))
            marg = reduce(logaddexp, Float32.(params(n)) .+ first.(cn))
            (marg, nothing)
        else
            @assert all(issomething.(last.(cn)))
            new_n = summate(last.(cn), reuse=n)
            new_n.log_probs = n.log_probs .+ first.(cn)
            (0.0f0, new_n)
        end
    end
    (marg, new_root) = foldup_aggregate(root, f_leaf, f_leaf, f_a, f_o, Tuple{Float32, Union{Nothing, Node}})

    @assert issomething(new_root)
    if marg != 0.0f0
        new_root = PlainSumNode([new_root])
        new_root.log_probs = [marg]
    end
    new_root
end

function forget(root, is_forgotten)
    (_, true_node) = canonical_constants(root)
    if isnothing(true_node)
        true_node = compile(typeof(root), true)
    end
    f_con(n) = n
    f_lit(n) = is_forgotten(variable(n)) ? true_node : n
    f_a(n, cn) = conjoin([cn...]; reuse=n) # convert type of cn
    f_o(n, cn) = disjoin([cn...]; reuse=n)
    foldup_aggregate(root, f_con, f_lit, f_a, f_o, Node)
end

"Update upper and lower bounds, and add log info"
function update_and_log(cur_root, root, quer, cache, lb, results, iter, post_prune; time=missing, prune_attempts=missing, split_var=missing)
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

    # Recompute upper and lower bounds
    ub = forward_bounds(cur_root, quer, cache)
    mstate, _ = max_sum_lower_bound(cur_root, quer, cache)
    if lb > MAR(root, mstate)[1] 
        println("Lower bound worsened from $(lb) to $(MAR(root, mstate)[1])")
    end
    new_lb = max(lb, MAR(root, mstate)[1]) # use the original circuit (before pruning) for lower bounds
    results[string("ub",postfix)][iter] = ub
    results[string("lb",postfix)][iter] = new_lb

    or_nodes = ⋁_nodes(cur_root)
    num_max = count(n -> associated_with_mult(n, quer, cache.impl_lits), or_nodes)
    results[string("num_edge",postfix)][iter] = num_edges(cur_root)
    results[string("num_node",postfix)][iter] = num_nodes(cur_root)
    results[string("num_sum",postfix)][iter] = length(or_nodes) - num_max
    results[string("num_max",postfix)][iter] = num_max

    ub, new_lb
end

"Compute marginal MAP by iteratively pruning and splitting"
function mmap_solve(root, quer; num_iter=length(quer), prune_attempts=10, log_per_iter=noop, heur="maxP")
    # initialize log
    num_iter = min(num_iter, length(quer))
    results = Dict()
    for x in MMAP_LOG_HEADER
        results[x] = Vector{Union{Any,Missing}}(missing,num_iter)
    end

    # marginalize out non-query variables
    # cur_root = root
    ub = forward_bounds(root, quer)
    _, mp = max_sum_lower_bound(root, quer)
    @show ub, log(mp)
    cur_root = marginalize_out(root, setdiff(variables(root), quer))
    @assert variables(cur_root) == quer
    # TODO: check the bounds before and after

    splittable = copy(quer)
    cache = MMAPCache()
    ub = forward_bounds(cur_root, quer, cache)
    _, mp = max_sum_lower_bound(cur_root, quer, cache)
    lb = log(mp)
    @show ub, lb
    counters = counter(Int)
    for i in 1:num_iter
        try
            if isempty(splittable) || ub < lb + 1e-10
                break
            end

            # Prune -- could improve the upper bound (TODO: maybe also the lower bound?)
            println("* Starting with $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
            tic = time_ns()
            # cur_root, prune_counters, actual_reps = prune(cur_root, quer, cache, exp(lb), prune_attempts)
            cur_root, prune_counters, actual_reps = prune(cur_root, quer, cache, lb, prune_attempts)
            merge!(counters, prune_counters)
            toc = time_ns()
            println("* Pruning gives $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
            @show update_and_log(cur_root,root,quer,cache,lb,results,i,true, prune_attempts=actual_reps, time=(toc-tic)/1.0e9)

            # Split root (move a quer variable up) -- could improve both the upper and lower bounds
            tic = time_ns()
            to_split = get_to_split(cur_root, splittable, counters, heur, lb)
            cur_root = add_and_split(cur_root, to_split)
            toc = time_ns()
            println("* Splitting on $(to_split) gives $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
            delete!(splittable, to_split)
            @show ub, lb = update_and_log(cur_root,root,quer,cache,lb,results,i,false, split_var=to_split, time=(toc-tic)/1.0e9)

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