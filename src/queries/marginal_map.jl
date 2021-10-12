export MMAP_LOG_HEADER, forward_bounds, edge_bounds, max_sum_lower_bound, associated_with_mult,
    mmap_solve, add_and_split, get_to_split, prune,
    mmap_reduced_mpe_state, gen_edges, brute_force_mmap, pc_condition, get_margs,
    marginalize_out, pc_condition

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
LitCacheDict = Dict{ProbCircuit, Union{BitSet, Nothing}}

struct MMAPCache
    ub::Dict{ProbCircuit, Float32}  # cache for upper bound forward pass
    lb::Dict{ProbCircuit,Tuple{Float32,Bool}}  # cache for lower bound forward pass (max-sum)
    impl_lits::LitCacheDict         # implied literals
    max_dec_node::Dict{ProbCircuit,Bool}
    variables::BitSet
    max_var::Var
end

MMAPCache(root) = begin
    vars = variables(root)
    MMAPCache(
    Dict{ProbCircuit, Float32}(), 
    Dict{ProbCircuit,Tuple{Float32,Bool}}(), 
    LitCacheDict(),
    Dict{ProbCircuit,Bool}(),
    vars, maximum(vars))
end

function update_lit_and_max_dec(root, query_vars, cache)
    implied_literals(root, cache.impl_lits)
    f_other(args...) = false
    f_o(n, cs) = begin
        # Check if a given sum node (with more than 2 children) is associated with any query variables
        if num_children(n) < 2
            return false
        end
    
        impl = [cache.impl_lits[x] for x in n.children]
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
    foldup_aggregate(root, f_other, f_other, f_other, f_o, Bool, cache.max_dec_node)
end

#####################
# Marginal MAP bound computations
##################### 

"Computes the upper bound on MMAP probability for each edge. Returns a map from node or edge to its upper bound"
function edge_bounds(root::ProbCircuit, query_vars::BitSet, thresh, cache::MMAPCache)
    update_lit_and_max_dec(root, query_vars, cache)
    forward_bounds(root, query_vars, cache)


    # tcache = NodeCacheDict(-Inf)
    # tcache[root] = 0.0
    # rcache = NodeCacheDict(-Inf)
    # rcache[root] = cache.ub[root]

    trcache = DefaultDict{ProbCircuit, Tuple{Float64, Float64}}((-Inf,-Inf))
    trcache[root] = (0.0, cache.ub[root])


    to_prune = Set{Tuple{ProbCircuit,ProbCircuit}}()

    lin = linearize(root, ProbCircuit)
    foreach(Iterators.reverse(lin)) do x 
        edge_bounds_fn(x, query_vars, thresh, to_prune, cache, trcache)
    end
    # @assert all(collect(values(rcache)) .>= 0.0)
    
    # Only keep in cache nodes that are not pruned in the previous iteration
    for k in setdiff(keys(cache.impl_lits), lin)
        delete!(cache.lb, k)
        delete!(cache.ub, k)
        delete!(cache.impl_lits, k)
    end

    to_prune
end

function edge_bounds_fn(root::ProbCircuit, query_vars::BitSet, thresh, to_prune, cache::MMAPCache, trcache)
    tcr, rcr = trcache[root]
    if isleaf(root) || tcr == -Inf
        return
    end
    cur = cache.ub[root]
    if is⋁gate(root)
        ismax = cache.max_dec_node[root]
        for (c, param) in zip(root.children, params(root))
            cuc = cache.ub[c]
            edge_val = if (ismax && cur - (param + cuc) > 1e-5)
                logsubexp(rcr, tcr + logsubexp(param + cuc, cur))
            else
                rcr
            end
            tcc, rcc = trcache[c]
            if cuc > -Inf
                edge_pr = param + tcr
                tcc = tcc > -Inf ? min(tcc, edge_pr) : edge_pr
            end
            rcc = max(rcc, edge_val)
            trcache[c] = (tcc, rcc)
            if edge_val < thresh - 1e-5
                push!(to_prune, (root, c)) 
            end
        end
    else
        for c in root.children
            tcc, rcc = trcache[c]
            rcc = max(rcc, rcr)
            tcc = tcc > -Inf ? min(tcc, tcr) : tcr
            trcache[c] = (tcc, rcc)
            if rcr < thresh - 1e-5
                push!(to_prune, (root, c)) 
            end
        end
    end
end

# TODO: perhaps a flag to skip recomputing impl_lits
function forward_bounds(root::ProbCircuit, query_vars::BitSet, cache) 
    update_lit_and_max_dec(root, query_vars, cache)

    f_leaf(_) = 0.0f0
    f_a(_, cs) = sum(cs)
    f_o(n, cs) = begin
        if cache.max_dec_node[n]
            maximum(Float32.(params(n)) .+ cs)
        else
            reduce(logaddexp, Float32.(params(n)) .+ cs)
        end
    end
    foldup_aggregate(root, f_leaf, f_leaf, f_a, f_o, Float32, cache.ub)
end

function forward_bounds(root::ProbCircuit, query_vars::BitSet, data::DataFrame, cache::MMAPCache)
    
    update_lit_and_max_dec(root, query_vars, cache)
    
    @assert num_features(data) == cache.max_var
    f_lit(n) = begin
        assignments = ispositive(n) ? data[:,variable(n)] : .!data[:,variable(n)]
        log.(Float32.(coalesce.(assignments, true)))
    end
    f_a(_, cs) = sum(cs)
    f_o(n, cs) = begin
        evals = map((p,cv) -> p .+ cv, Float32.(params(n)), cs)
        if cache.max_dec_node[n]
            reduce((x,y) -> max.(x,y), evals)
        else
            reduce((x,y) -> logaddexp.(x,y), evals)
        end
    end
    foldup_aggregate(root, f_lit, f_lit, f_a, f_o, Array{Float32})
end

"""
Compute the lower bound on MMAP using a max-sum circuit
 
I.e. a forward bound that takes sum until encountering a max node.
If the circuit has a constrained vtree for the query variables, returns the exact MMAP.
"""
function max_sum_lower_bound(root::ProbCircuit, query_vars::BitSet, cache)
    
    update_lit_and_max_dec(root, query_vars, cache)

    # forward pass 
    f_leaf(_) = (0.0f0, false)
    f_a(_, cs) = reduce((c1,c2) -> (c1[1]+c2[1], c1[2]||c2[2]), cs)
    f_o(n, cs) = begin
        has_max = any(last.(cs))    # whether n has a max node as a descendant
        if has_max || cache.max_dec_node[n]
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
    state = Array{Union{Nothing,Bool}}(nothing, cache.max_var)
    max_sum_down(root, cache.lb, state)

    # reduce to query variables
    @assert all(issomething.(state[collect(query_vars)]))
    # reduced = transpose([i in query_vars ? state[i] : missing for i in 1:num_variables(root)])
    reduced = transpose([i in query_vars ? state[i] : missing for i in 1:cache.max_var])
    df = DataFrame(reduced, :auto)
    df, exp(custom_MAR(root, reduced))
end

function custom_MAR(root, data)
    f_leaf(n) = if (data[variable(n)] === nothing ||
                    data[variable(n)] == ispositive(n))
            log(one(Float32))
        else
            log(zero(Float32))
        end 
    f_a(n, call) = mapreduce(call, +, n.children)
    f_o(n, call) = begin
        r = log(zero(Float32))
        for i = 1:length(n.children)
           r = logaddexp(r, Float32(n.log_probs[i]) + call(n.children[i])) 
        end
        r
    end
    foldup(root, f_leaf, f_leaf, f_a, f_o, Float32)
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
    counters = counter(Int)
    actual_reps = num_reps
    for i in 1:num_reps
        to_prune = find_to_prune(circ, query_vars, cache, lower_bound, counters)
        if !isempty(to_prune)
            circ = do_pruning(circ, to_prune, cache) 
        else # early terminate if no more edges are getting pruned
            actual_reps = i
            break
        end
    end
    circ, counters, actual_reps
end

"Find edges to prune using a given lower bound (thresh) on the MMAP probability"
function find_to_prune(n::ProbCircuit, query_vars::BitSet, cache::MMAPCache, thresh, counters)
    update_lit_and_max_dec(n, query_vars, cache)

    to_prune = edge_bounds(n, query_vars, thresh, cache)
    
    # Caution: if buffer is too small, edges may be pruned incorrectly. if it's too big, too little pruning may happen.

    # println("to_prune: $(length(to_prune))")

    query_lits = union(query_vars, BitSet(map(x->-x, collect(query_vars))))
    for (n,x) in to_prune
        decided_lits = setdiff(cache.impl_lits[x], cache.impl_lits[n])
        decided_query_lits = intersect(decided_lits, query_lits)
        merge!(counters, counter(decided_query_lits))
    end

    to_prune
end

"Perform the actual pruning. Note this will always return a plain logic circuit"
function do_pruning(root, to_prune, cache)
    
    f_leaf(n) = n
    f_inner(n, call) = begin
        # TODO optimize for the case where nothing gets deleted
        # Find children we are keeping
        inds = findall(x -> (n, x) ∉ to_prune, n.children)
        new_children = map(call, n.children[inds])
        if isempty(inds)
            nothing  # no need to create a new node if all children are pruned
        elseif new_children == n.children
            n        # reuse node if no edges below are pruned
        elseif is⋁gate(n)
            del_n = PlainSumNode(new_children)
            del_n.log_probs = n.log_probs[inds]
            del_n
        else
            PlainMulNode(new_children)
        end
    end

    foldup(root, f_leaf, f_inner, Union{Nothing,ProbCircuit})
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

    split_root = custom_split(root, var)

    # new_and = PlainMulNode([root])
    # new_root = PlainSumNode([new_and])
    # split_root = split(new_root, (new_root, new_and), Var(var), sanity_check=false, callback=keep_params, keep_unary=true)[1]
    
    split_root.log_probs .= 0   # each child has var as evidence and is left unnormalized
    split_root
end

function custom_split(root, var)

    f_leaf(n) = begin
        if variable(n) == var
            if ispositive(n)
                (n, nothing)
            else
                (nothing, n)
            end
        else
            (n,n)
        end
    end

    f_a(n, cn) = begin 
        left_changed = false
        left_false = false
        right_changed = false
        right_false = false
        for i = 1:length(cn)
            !left_false && isnothing(cn[i][1]) && (left_false = true)
            !right_false && isnothing(cn[i][2]) && (right_false = true)
            !left_changed && cn[i][1] !== n.children[i] && (left_changed = true)
            !right_changed && cn[i][2] !== n.children[i] && (right_changed = true)
        end

        left = if left_false
            nothing
        elseif length(cn) == 1
            cn[1][1]
        elseif !left_changed
            n
        else
            PlainMulNode(first.(cn))
        end

        right = if right_false
            nothing
        elseif length(cn) == 1
            cn[1][2]
        elseif !right_changed
            n
        else   
            PlainMulNode(last.(cn))
        end
        
        # @assert isnothing(left) || LogicCircuits.num_children(left) > 1
        # @assert isnothing(right) || LogicCircuits.num_children(right) > 1

        (left, right)
    end

    f_o(n, cn) = begin
        
        left_changed = false
        left_false = true
        right_changed = false
        right_false = true
        for i = 1:length(cn)
            left_false && issomething(cn[i][1]) && (left_false = false)
            right_false && issomething(cn[i][2]) && (right_false = false)
            !left_changed && cn[i][1] !== n.children[i] && (left_changed = true)
            !right_changed && cn[i][2] !== n.children[i] && (right_changed = true)
        end

        left = if left_false
            nothing
        elseif !left_changed
            n
        else            
            leftcn = ProbCircuit[]
            leftpa = Float64[]
            for i = 1:length(cn)
                if issomething(cn[i][1])
                    push!(leftcn, cn[i][1])
                    push!(leftpa, n.log_probs[i])
                end
            end
            left = PlainSumNode(leftcn)
            left.log_probs .= leftpa
            left
        end

        right = if right_false
            nothing
        elseif !right_changed
            n
        else            
            rightcn = ProbCircuit[]
            rightpa = Float64[]
            for i = 1:length(cn)
                if issomething(cn[i][2])
                    push!(rightcn, cn[i][2])
                    push!(rightpa, n.log_probs[i])
                end
            end
            right = PlainSumNode(rightcn)
            right.log_probs .= rightpa
            right
        end

        (left, right)
    end

    T = Tuple{Union{Nothing,ProbCircuit},Union{Nothing,ProbCircuit}}
    l,r = foldup_aggregate(root, f_leaf, f_leaf, f_a, f_o, T)

    PlainSumNode([l,r])
end

"Function to pass to condition so that it maintains parameters"
function keep_params(new_n, n, kept)
    new_n.log_probs = n.log_probs[kept]
end


function get_to_split(root, splittable, counters, heur, lb, cache::MMAPCache)    
    if heur == "avgUB" || heur == "minUB" || heur == "maxUB" || heur == "UB"
        vars = collect(splittable)
        datamat = Array{Union{Missing, Bool}}(missing, 2*length(vars), cache.max_var)
        for i in 1:length(vars)
            datamat[2*i-1, vars[i]] = true
            datamat[2*i, vars[i]] = false
        end
        bounds = forward_bounds(root, splittable, DataFrame(datamat, :auto), cache)
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

"Update upper and lower bounds"
function update_bounds(cur_root, root, quer, cache, lb, lb_state)
    ub = forward_bounds(cur_root, quer, cache)
    mstate, _ = max_sum_lower_bound(cur_root, quer, cache)
    if !isequal(mstate, lb_state)
        mp = MAR(root, mstate)[1]  # use the original circuit (before pruning) for lower bounds
        if mp > lb
            lb = mp
            lb_state = mstate
        end
    end
    return ub, lb, lb_state
end

"Update log info and call log function if specified"
function update_and_log(cur_root, quer, timeout, cache, ub, lb, results, iter, time, post_prune; callback=noop, prune_attempts=missing, split_var=missing)
    if post_prune
        results["iter"][iter] = iter
        results["prune_time"][iter] = time
        results["prune_attempts"][iter] = prune_attempts
        postfix = "_post_prune"
    else # post_split
        results["split_time"][iter] = time
        results["total_time"][iter] = total_time = (iter==1 ? 0 : results["total_time"][iter-1]) + results["prune_time"][iter] + time
        results["split_var"][iter] = split_var
        postfix = "_post_split"
    end

    results[string("ub",postfix)][iter] = ub
    results[string("lb",postfix)][iter] = lb

    or_nodes = ⋁_nodes(cur_root)
    num_max = count(n -> cache.max_dec_node[n], or_nodes)
    results[string("num_edge",postfix)][iter] = num_edges(cur_root)
    results[string("num_node",postfix)][iter] = num_nodes(cur_root)
    results[string("num_sum",postfix)][iter] = length(or_nodes) - num_max
    results[string("num_max",postfix)][iter] = num_max

    if !post_prune
        callback(results)
        if total_time > timeout
            error("timeout")
        end
    end
end

"Compute marginal MAP by iteratively pruning and splitting"
function mmap_solve(root, quer; num_iter=length(quer), prune_attempts=10, log_per_iter=noop, heur="maxP", timeout=3600, verbose=false, out=stdout)
    # initialize log
    num_iter = min(num_iter, length(quer))
    results = Dict()
    for x in MMAP_LOG_HEADER
        results[x] = Vector{Union{Any,Missing}}(missing,num_iter)
    end

    # marginalize out non-query variables
    # cur_root = root
    # ub = forward_bounds(root, quer)
    # _, mp = max_sum_lower_bound(root, quer)
    cur_root = marginalize_out(root, setdiff(variables(root), quer))
    @assert variables(cur_root) == quer
    # TODO: check the bounds before and after

    splittable = copy(quer)
    cache = MMAPCache(cur_root)
    ub = forward_bounds(cur_root, quer, cache)
    lb_state, mp = max_sum_lower_bound(cur_root, quer, cache)
    lb = log(mp)
    # @show ub, lb
    counters = counter(Int)
    for i in 1:num_iter
        try
            if isempty(splittable) || ub < lb + 1e-10
                break
            end

            # Prune -- could improve the upper bound (TODO: maybe also the lower bound?)
            verbose && println(out, "* Starting with $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
            tic = time_ns()
            # cur_root, prune_counters, actual_reps = prune(cur_root, quer, cache, exp(lb), prune_attempts)
            cur_root, prune_counters, actual_reps = prune(cur_root, quer, cache, lb, prune_attempts)
            merge!(counters, prune_counters)
            ub, lb, lb_state = update_bounds(cur_root, root, quer, cache, lb, lb_state)
            toc = time_ns()
            verbose && println(out, "* Pruning gives $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
            verbose && update_and_log(cur_root,quer,timeout,cache,ub,lb,results,i,(toc-tic)/1.0e9,true, prune_attempts=actual_reps)

            # Split root (move a quer variable up) -- could improve both the upper and lower bounds
            tic = time_ns()
            to_split = get_to_split(cur_root, splittable, counters, heur, lb, cache)
            cur_root = add_and_split(cur_root, to_split)
            ub, lb, lb_state = update_bounds(cur_root, root, quer, cache, lb, lb_state)
            toc = time_ns()
            delete!(splittable, to_split)
            verbose && println(out, "* Splitting on $(to_split) gives $(num_edges(cur_root)) edges and $(num_nodes(cur_root)) nodes.")
            verbose && update_and_log(cur_root,quer,timeout,cache,ub,lb,results,i,(toc-tic)/1.0e9,false, split_var=to_split, callback=log_per_iter)
        catch e
            println(out, sprint(showerror, e, backtrace()))
            break
        end
    end
    # TODO: return results (timeout, iter, time, ub, lb, lbstate)
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

pc_condition(root::PlainProbCircuit, lit1, lit2, lit3...) = pc_condition(pc_condition(root, lit1), lit2, lit3...)

function pc_condition(root::PlainProbCircuit, lit)
    conjoin(root, lit, callback=keep_params, keep_unary=true)
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