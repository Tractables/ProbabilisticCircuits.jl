export forward_bounds, edge_bounds

using LogicCircuits
using StatsFuns: logaddexp
using DataStructures: DefaultDict

#####################
# Circuit Marginal Map
##################### 

# Everything to do with search based marginal map computation

# 
function edge_bounds(root::ProbCircuit, query_vars::BitSet)
    impl_lits = implied_literals(root)
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
            if num_children(root) == 2 && associated_with(root, query_vars, impl_lits)
                @show rcache[(root, c)] = rcache[root] + tcache[root] * (exp(param) * exp(mcache[c]) - exp(mcache[root]))
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
    impl_lits = implied_literals(root)
    forward_bounds_rec(root, query_vars, Dict{ProbCircuit, Float32}(), impl_lits)
end


function forward_bounds_rec(root::ProbCircuit, query_vars::BitSet, mcache::Dict{ProbCircuit, Float32}, impl_lits)
    if isleaf(root)
        mcache[root] = 0.0
    elseif isinner(root)
        for c in root.children
            forward_bounds_rec(c, query_vars, mcache, impl_lits)
        end
        if is⋀gate(root)
            mcache[root] = mapreduce(c -> mcache[c], +, root.children)
        else
            @assert(num_children(root) <= 2)
            # If we have just the one child, just incorporate the parameter
            if num_children(root) == 1
                mcache[root] = mcache[root.children[1]] + params(root)[1]
            else
                # If we have 2 children, check if associated:
                if associated_with(root, query_vars, impl_lits)
                    # If it is, we're taking a max
                    
                    mcache[root] = mapreduce((c,p) -> mcache[c] + p, max, root.children, params(root))
                else
                    # If it isn't, we're taking a sum
                    mcache[root] = mapreduce((c,p) -> mcache[c] + p, logaddexp, root.children, params(root))
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