export pr_constraint

const PRCache = Dict{Tuple{ProbCircuit, LogicCircuit}, Float64}

# Arthur Choi, Guy Van den Broeck, and Adnan Darwiche. Tractable learning for structured probability
# spaces: A case study in learning preference distributions. In Proceedings of IJCAI, 2015.

"""
Calculate the probability of the logic formula given by LC for the PC
"""
function pr_constraint(pc_node::StructProbCircuit, lc_node, cache::PRCache=PRCache())::Float64

    # TODO require that both circuits have an equal vtree for safety. If they don't, then first convert them to have a vtree
    @assert respects_vtree(lc_node, vtree(pc_node)) "Both circuits do not have an equal vtree"

    # Cache hit
    if (pc_node, lc_node) in keys(cache) 
        return cache[pc_node, lc_node]
    
    # Boundary cases
    elseif isliteralgate(pc_node)
        # Both are literals, just check whether they agrees with each other 
        if isliteralgate(lc_node)
            if literal(pc_node) == literal(lc_node)
                return get!(cache, (pc_node, lc_node), 1.0)
            else
                return get!(cache, (pc_node, lc_node), 0.0)
            end
        else
            pr_constraint(pc_node, children(lc_node)[1], cache)
            if length(children(lc_node)) > 1
                pr_constraint(pc_node, children(lc_node)[2], cache)
                return get!(cache, (pc_node, lc_node), 1.0)
            else
                return get!(cache, (pc_node, lc_node),
                    literal(children(lc_node)[1]) == literal(pc_node) ? 1.0 : 0.0)
            end
        end
    
    # The pc is true
    elseif isliteralgate(children(pc_node)[1])
        theta = exp(pc_node.log_probs[1])
        return get!(cache, (pc_node, lc_node),
            theta * pr_constraint(children(pc_node)[1], lc_node, cache) +
            (1.0 - theta) * pr_constraint(children(pc_node)[2], lc_node, cache))
    
    # Both pcs are not trivial
    else 
        prob = 0.0
        for (prob⋀_node, log_theta) in zip(children(pc_node), pc_node.log_probs)
            p = children(prob⋀_node)[1]
            s = children(prob⋀_node)[2]

            theta = exp(log_theta)
            for lc⋀_node in children(lc_node)
                r = children(lc⋀_node)[1]
                t = children(lc⋀_node)[2]
                prob += theta * pr_constraint(p, r, cache) * pr_constraint(s, t, cache)
            end
        end
        return get!(cache, (pc_node, lc_node), prob)
    end
end