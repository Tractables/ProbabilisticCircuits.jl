using DataStructures

# Arthur Choi, Guy Van den Broeck, and Adnan Darwiche. Tractable learning for structured probability
# spaces: A case study in learning preference distributions. In Proceedings of IJCAI, 2015.
"Calculate the probability of the logic formula given by sdd for the psdd"
function pr_constraint(psdd_node::ProbNode, sdd_node::Union{ProbNode, StructLogicCircuit})
    cache = Dict{Tuple{ProbNode, Union{ProbNode, StructLogicCircuit}}, Float64}()

    return pr_constraint(psdd_node, sdd_node, cache)
end
function pr_constraint(psdd_node::ProbNode, sdd_node::Union{ProbNode, StructLogicCircuit},
                       cache::Dict{Tuple{ProbNode, Union{ProbNode, StructLogicCircuit}}, Float64})::Float64
    if (psdd_node, sdd_node) in keys(cache) # Cache hit
        return cache[psdd_node, sdd_node]
    elseif psdd_node isa ProbLiteral # Boundary cases
        if sdd_node isa Union{ProbLiteral, PlainStructLiteralNode} # Both are literals, just check whether they agrees with each other
            if literal(psdd_node) == literal(sdd_node)
                return get!(cache, (psdd_node, sdd_node), 1.0)
            else
                return get!(cache, (psdd_node, sdd_node), 0.0)
            end
        else
            pr_constraint(psdd_node, sdd_node.children[1], cache)
            if length(sdd_node.children) > 1
                pr_constraint(psdd_node, sdd_node.children[2], cache)
                return get!(cache, (psdd_node, sdd_node), 1.0)
            else
                return get!(cache, (psdd_node, sdd_node),
                    literal(sdd_node.children[1]) == literal(psdd_node) ? 1.0 : 0.0
                )
            end
        end
    elseif psdd_node.children[1] isa ProbLiteral # The psdd is true
        theta = exp(psdd_node.log_thetas[1])
        return get!(cache, (psdd_node, sdd_node),
            theta * pr_constraint(psdd_node.children[1], sdd_node, cache) +
            (1.0 - theta) * pr_constraint(psdd_node.children[2], sdd_node, cache)
        )
    else # Both psdds are not trivial
        prob = 0.0
        for (prob⋀_node, log_theta) in zip(psdd_node.children, psdd_node.log_thetas)
            p = prob⋀_node.children[1]
            s = prob⋀_node.children[2]

            theta = exp(log_theta)
            for sdd⋀_node in sdd_node.children
                r = sdd⋀_node.children[1]
                t = sdd⋀_node.children[2]
                prob += theta * pr_constraint(p, r, cache) * pr_constraint(s, t, cache)
            end
        end
        return get!(cache, (psdd_node, sdd_node), prob)
    end
end


"Entropy of the distribution of the input psdd."
function psdd_entropy(psdd_node::ProbNode)::Float64
    psdd_entropy_cache = Dict{ProbNode, Float64}()

    return psdd_entropy(psdd_node, psdd_entropy_cache)
end
function psdd_entropy(psdd_node::Prob⋁, psdd_entropy_cache::Dict{ProbNode, Float64})::Float64
    if psdd_node in keys(psdd_entropy_cache)
        return psdd_entropy_cache[psdd_node]
    elseif psdd_node.children[1] isa ProbLiteral
        return get!(psdd_entropy_cache, psdd_node,
            - exp(psdd_node.log_thetas[1]) * psdd_node.log_thetas[1] -
            exp(psdd_node.log_thetas[2]) * psdd_node.log_thetas[2]
        )
    else
        local_entropy = 0.0
        for (prob⋀_node, log_prob) in zip(psdd_node.children, psdd_node.log_thetas)
            p = prob⋀_node.children[1]
            s = prob⋀_node.children[2]

            local_entropy += exp(log_prob) * (psdd_entropy(p, psdd_entropy_cache) +
                psdd_entropy(s, psdd_entropy_cache) - log_prob)
        end
        return get!(psdd_entropy_cache, psdd_node, local_entropy)
    end
end
function psdd_entropy(psdd_node::Prob⋀, psdd_entropy_cache::Dict{ProbNode, Float64})::Float64
    return get!(psdd_entropy_cache, psdd_node.children[1], psdd_entropy(psdd_node.children[1], psdd_entropy_cache)) +
        get!(psdd_entropy_cache, psdd_node.children[2], psdd_entropy(psdd_node.children[2], psdd_entropy_cache))
end
function psdd_entropy(psdd_node::ProbLiteral, psdd_entropy_cache::Dict{ProbNode, Float64})::Float64
    return get!(psdd_entropy_cache, psdd_node, 0.0)
end


"KL divergence calculation for psdds that are not necessarily identical"
function psdd_kl_divergence(psdd_node1::ProbNode, psdd_node2::ProbNode)::Float64
    kl_divergence_cache = Dict{Tuple{ProbNode, ProbNode}, Float64}()
    pr_constraint_cache = Dict{Tuple{ProbNode, Union{ProbNode, StructLogicCircuit}}, Float64}()

    return psdd_kl_divergence(psdd_node1, psdd_node2, kl_divergence_cache, pr_constraint_cache)
end
function psdd_kl_divergence(psdd_node1::ProbNode, psdd_node2::ProbNode,
        kl_divergence_cache::Dict{Tuple{ProbNode, ProbNode}, Float64})::Float64
    pr_constraint_cache = Dict{Tuple{ProbNode, Union{ProbNode, StructLogicCircuit}}, Float64}()

    return psdd_kl_divergence(psdd_node1, psdd_node2, kl_divergence_cache, pr_constraint_cache)
end
function psdd_kl_divergence(psdd_node1::ProbNode, psdd_node2::ProbNode,
        kl_divergence_cache::Dict{Tuple{ProbNode, ProbNode}, Float64},
        pr_constraint_cache::Dict{Tuple{ProbNode, Union{ProbNode, StructLogicCircuit}}, Float64})
    @assert !(psdd_node1 isa Prob⋀ || psdd_node2 isa Prob⋀) "Prob⋀ not a valid PSDD node for KL-Divergence"

    # Check if both nodes are normalized for same vtree node
    @assert variables(psdd_node1.origin.vtree) == variables(psdd_node2.origin.vtree) "Both nodes not normalized for same vtree node"

    if (psdd_node1, psdd_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[(psdd_node1, psdd_node2)]
    elseif psdd_node1.children[1] isa ProbLiteral
        if psdd_node2 isa ProbLiteral
            psdd_kl_divergence(psdd_node1.children[1], psdd_node2, kl_divergence_cache, pr_constraint_cache)
            psdd_kl_divergence(psdd_node1.children[2], psdd_node2, kl_divergence_cache, pr_constraint_cache)
            if literal(psdd_node1.children[1]) == literal(psdd_node2)
                return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                    psdd_node1.log_thetas[1] * exp(psdd_node1.log_thetas[1])
                )
            else
                return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                    psdd_node1.log_thetas[2] * exp(psdd_node1.log_thetas[2])
                )
            end
        else
            # The below four lines actually assign zero, but still we need to
            # call it.
            psdd_kl_divergence(psdd_node1.children[1], psdd_node2.children[1], kl_divergence_cache, pr_constraint_cache)
            psdd_kl_divergence(psdd_node1.children[1], psdd_node2.children[2], kl_divergence_cache, pr_constraint_cache)
            psdd_kl_divergence(psdd_node1.children[2], psdd_node2.children[1], kl_divergence_cache, pr_constraint_cache)
            psdd_kl_divergence(psdd_node1.children[2], psdd_node2.children[2], kl_divergence_cache, pr_constraint_cache)
            # There are two possible matches
            if literal(psdd_node1.children[1]) == literal(psdd_node2.children[1])
                return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                    exp(psdd_node1.log_thetas[1]) * (psdd_node1.log_thetas[1] - psdd_node2.log_thetas[1]) +
                    exp(psdd_node1.log_thetas[2]) * (psdd_node1.log_thetas[2] - psdd_node2.log_thetas[2])
                )
            else
                return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                    exp(psdd_node1.log_thetas[1]) * (psdd_node1.log_thetas[1] - psdd_node2.log_thetas[2]) +
                    exp(psdd_node1.log_thetas[2]) * (psdd_node1.log_thetas[2] - psdd_node2.log_thetas[1])
                )
            end
        end
    else # the normal case
        kl_divergence = 0.0

        # loop through every combination of prim and sub
        for (prob⋀_node1, log_theta1) in zip(psdd_node1.children, psdd_node1.log_thetas)
            for (prob⋀_node2, log_theta2) in zip(psdd_node2.children, psdd_node2.log_thetas)
                p = prob⋀_node1.children[1]
                s = prob⋀_node1.children[2]

                r = prob⋀_node2.children[1]
                t = prob⋀_node2.children[2]

                theta1 = exp(log_theta1)

                p11 = pr_constraint(s, t, pr_constraint_cache)
                p12 = pr_constraint(p, r, pr_constraint_cache)

                p13 = theta1 * (log_theta1 - log_theta2)

                p21 = psdd_kl_divergence(p, r, kl_divergence_cache, pr_constraint_cache)
                p31 = psdd_kl_divergence(s, t, kl_divergence_cache, pr_constraint_cache)

                kl_divergence += p11 * p12 * p13 + theta1 * (p11 * p21 + p12 * p31)
            end
        end
        return get!(kl_divergence_cache, (psdd_node1, psdd_node2), kl_divergence)
    end
end
function psdd_kl_divergence(psdd_node1::ProbLiteral, psdd_node2::ProbLiteral,
        kl_divergence_cache::Dict{Tuple{ProbNode, ProbNode}, Float64},
        pr_constraint_cache::Dict{Tuple{ProbNode, Union{ProbNode, StructLogicCircuit}}, Float64})
    # Check if literals are over same variables in vtree
   @assert variables(psdd_node1.origin.vtree) == variables(psdd_node2.origin.vtree) "Both nodes not normalized for same vtree node"

    if (psdd_node1, psdd_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[psdd_node1, psdd_node2]
    else
        # In this case probability is 1, kl divergence is 0
        return get!(kl_divergence_cache, (psdd_node1, psdd_node2), 0.0)
    end
end
function psdd_kl_divergence(psdd_node1::Prob⋁, psdd_node2::ProbLiteral,
        kl_divergence_cache::Dict{Tuple{ProbNode, ProbNode}, Float64},
        pr_constraint_cache::Dict{Tuple{ProbNode, Union{ProbNode, StructLogicCircuit}}, Float64})
    @assert variables(psdd_node1.origin.vtree) == variables(psdd_node2.origin.vtree) "Both nodes not normalized for same vtree node"

    if (psdd_node1, psdd_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[psdd_node1, psdd_node2]
    else
        psdd_kl_divergence(psdd_node1.children[1], psdd_node2, kl_divergence_cache, pr_constraint_cache)
        psdd_kl_divergence(psdd_node1.children[2], psdd_node2, kl_divergence_cache, pr_constraint_cache)
        if literal(psdd_node1.children[1]) == literal(psdd_node2)
            return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                psdd_node1.log_thetas[1] * exp(psdd_node1.log_thetas[1])
            )
        else
            return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                psdd_node1.log_thetas[2] * exp(psdd_node1.log_thetas[2])
            )
        end
    end
end
function psdd_kl_divergence(psdd_node1::ProbLiteral, psdd_node2::Prob⋁,
        kl_divergence_cache::Dict{Tuple{ProbNode, ProbNode}, Float64},
        pr_constraint_cache::Dict{Tuple{ProbNode, Union{ProbNode, StructLogicCircuit}}, Float64})
    @assert variables(psdd_node1.origin.vtree) == variables(psdd_node2.origin.vtree) "Both nodes not normalized for same vtree node"

    if (psdd_node1, psdd_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[psdd_node1, psdd_node2]
    else
        psdd_kl_divergence(psdd_node1, psdd_node2.children[1], kl_divergence_cache, pr_constraint_cache)
        psdd_kl_divergence(psdd_node1, psdd_node2.children[2], kl_divergence_cache, pr_constraint_cache)
        if literal(psdd_node1) == literal(psdd_node2.children[1])
            return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                -psdd_node2.log_thetas[1]
            )
        else
            return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                -psdd_node2.log_thetas[2]
            )
        end
    end
end
