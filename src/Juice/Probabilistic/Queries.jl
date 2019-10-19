
# Arthur Choi, Guy Van den Broeck, and Adnan Darwiche. Tractable learning for structured probability
# spaces: A case study in learning preference distributions. In Proceedings of IJCAI, 2015.
"Calculate the probability of the logic formula given by sdd for the psdd"
function pr_constraint(psdd::ProbΔNode, sdd::Union{ProbΔNode, StructLogicalΔNode})
    cache = Dict{Tuple{ProbΔNode, Union{ProbΔNode, StructLogicalΔNode}}, Float64}()

    return pr_constraint(psdd, sdd, cache)
end
function pr_constraint(psdd::ProbΔNode, sdd::Union{ProbΔNode, StructLogicalΔNode},
                       cache::Dict{Tuple{ProbΔNode, Union{ProbΔNode, StructLogicalΔNode}}, Float64})::Float64
    if (psdd, sdd) in keys(cache) # Cache hit
        return cache[(psdd, sdd)]
    elseif psdd isa ProbLiteral # Boundary cases
        if sdd isa Union{ProbLiteral, StructLiteralNode} # Both are literals, just check whether they agrees with each other
            if literal(psdd) == literal(sdd)
                return get!(cache, (psdd, sdd), 1.0)
            else
                return get!(cache, (psdd, sdd), 0.0)
            end
        else
            pr_constraint(psdd, sdd.children[1], cache)
            pr_constraint(psdd, sdd.children[2], cache)
            return get!(cache, (psdd, sdd), 1.0)
        end
    elseif psdd.children[1] isa ProbLiteral # The psdd is true
        theta = exp(psdd.log_thetas[1])
        return get!(cache, (psdd, sdd),
            theta * pr_constraint(psdd.children[1], sdd, cache) + (1.0 - theta) * pr_constraint(psdd.children[2], sdd, cache)
        )
    else # Both psdds are not trivial
        prob = 0.0
        for (prob⋀_node, log_theta) in zip(psdd.children, psdd.log_thetas)
            p = prob⋀_node.children[1]
            s = prob⋀_node.children[2]

            theta = exp(log_theta)
            for sdd⋀_node in sdd.children
                r = sdd⋀_node.children[1]
                t = sdd⋀_node.children[2]
                prob += theta * pr_constraint(p, r, cache) * pr_constraint(s, t, cache)
            end
        end
        return get!(cache, (psdd, sdd), prob)
    end
end
