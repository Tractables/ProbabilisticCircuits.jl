export kl_divergence, entropy

const KLDCache = Dict{Tuple{ProbCircuit,ProbCircuit}, Float64}

""""
Calculate entropy of the distribution of the input pc."
"""

function entropy(pc_node::StructSumNode, pc_entropy_cache::Dict{ProbCircuit, Float64}=Dict{ProbCircuit, Float64}())::Float64
    if pc_node in keys(pc_entropy_cache)
        return pc_entropy_cache[pc_node]
    elseif children(pc_node)[1] isa StructProbLiteralNode
        return get!(pc_entropy_cache, pc_node,
            - exp(pc_node.log_probs[1]) * pc_node.log_probs[1] -
            exp(pc_node.log_probs[2]) * pc_node.log_probs[2])
    else
        local_entropy = 0.0
        for (prob⋀_node, log_prob) in zip(children(pc_node), pc_node.log_probs)
            p = children(prob⋀_node)[1]
            s = children(prob⋀_node)[2]

            local_entropy += exp(log_prob) * (entropy(p, pc_entropy_cache) +
                entropy(s, pc_entropy_cache) - log_prob)
        end
        return get!(pc_entropy_cache, pc_node, local_entropy)
    end
end

function entropy(pc_node::StructMulNode, pc_entropy_cache::Dict{ProbCircuit, Float64})::Float64
    return get!(pc_entropy_cache, children(pc_node)[1], entropy(children(pc_node)[1], pc_entropy_cache)) +
        get!(pc_entropy_cache, children(pc_node)[2], entropy(children(pc_node)[2], pc_entropy_cache))
end

function entropy(pc_node::StructProbLiteralNode, pc_entropy_cache::Dict{ProbCircuit, Float64})::Float64
    return get!(pc_entropy_cache, pc_node, 0.0)
end

"Calculate KL divergence calculation for pcs that are not necessarily identical"
function kl_divergence(pc_node1::StructSumNode, pc_node2::StructSumNode,
        kl_divergence_cache::KLDCache=KLDCache(), pr_constraint_cache::PRCache=PRCache())
    @assert !(pc_node1 isa StructMulNode || pc_node2 isa StructMulNode) "Prob⋀ not a valid pc node for KL-Divergence"

    # Check if both nodes are normalized for same vtree node
    @assert variables(pc_node1) == variables(pc_node2) "Both nodes not normalized for same vtree node"

    if (pc_node1, pc_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[(pc_node1, pc_node2)]
    elseif children(pc_node1)[1] isa StructProbLiteralNode
        if pc_node2 isa StructProbLiteralNode
            kl_divergence(children(pc_node1)[1], pc_node2, kl_divergence_cache, pr_constraint_cache)
            kl_divergence(children(pc_node1)[2], pc_node2, kl_divergence_cache, pr_constraint_cache)
            if literal(children(pc_node1)[1]) == literal(pc_node2)
                return get!(kl_divergence_cache, (pc_node1, pc_node2),
                    pc_node1.log_probs[1] * exp(pc_node1.log_probs[1])
                )
            else
                return get!(kl_divergence_cache, (pc_node1, pc_node2),
                    pc_node1.log_probs[2] * exp(pc_node1.log_probs[2])
                )
            end
        else
            # The below four lines actually assign zero, but still we need to
            # call it.
            kl_divergence(children(pc_node1)[1], children(pc_node2)[1], kl_divergence_cache, pr_constraint_cache)
            kl_divergence(children(pc_node1)[1], children(pc_node2)[2], kl_divergence_cache, pr_constraint_cache)
            kl_divergence(children(pc_node1)[2], children(pc_node2)[1], kl_divergence_cache, pr_constraint_cache)
            kl_divergence(children(pc_node1)[2], children(pc_node2)[2], kl_divergence_cache, pr_constraint_cache)
            # There are two possible matches
            if literal(children(pc_node1)[1]) == literal(children(pc_node2)[1])
                return get!(kl_divergence_cache, (pc_node1, pc_node2),
                    exp(pc_node1.log_probs[1]) * (pc_node1.log_probs[1] - pc_node2.log_probs[1]) +
                    exp(pc_node1.log_probs[2]) * (pc_node1.log_probs[2] - pc_node2.log_probs[2])
                )
            else
                return get!(kl_divergence_cache, (pc_node1, pc_node2),
                    exp(pc_node1.log_probs[1]) * (pc_node1.log_probs[1] - pc_node2.log_probs[2]) +
                    exp(pc_node1.log_probs[2]) * (pc_node1.log_probs[2] - pc_node2.log_probs[1])
                )
            end
        end
    else # the normal case
        kld = 0.0

        # loop through every combination of prim and sub
        for (prob⋀_node1, log_theta1) in zip(children(pc_node1), pc_node1.log_probs)
            for (prob⋀_node2, log_theta2) in zip(children(pc_node2), pc_node2.log_probs)
                p = children(prob⋀_node1)[1]
                s = children(prob⋀_node1)[2]

                r = children(prob⋀_node2)[1]
                t = children(prob⋀_node2)[2]

                theta1 = exp(log_theta1)

                p11 = pr_constraint(s, t, pr_constraint_cache)
                p12 = pr_constraint(p, r, pr_constraint_cache)

                p13 = theta1 * (log_theta1 - log_theta2)

                p21 = kl_divergence(p, r, kl_divergence_cache, pr_constraint_cache)
                p31 = kl_divergence(s, t, kl_divergence_cache, pr_constraint_cache)

                kld += p11 * p12 * p13 + theta1 * (p11 * p21 + p12 * p31)
            end
        end
        return get!(kl_divergence_cache, (pc_node1, pc_node2), kld)
    end
end

function kl_divergence(pc_node1::StructProbLiteralNode, pc_node2::StructProbLiteralNode,
        kl_divergence_cache::KLDCache, pr_constraint_cache::PRCache)
    # Check if literals are over same variables in vtree
   @assert variables(pc_node1) == variables(pc_node2) "Both nodes not normalized for same vtree node"

    if (pc_node1, pc_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[pc_node1, pc_node2]
    else
        # In this case probability is 1, kl divergence is 0
        return get!(kl_divergence_cache, (pc_node1, pc_node2), 0.0)
    end
end

function kl_divergence(pc_node1::StructSumNode, pc_node2::StructProbLiteralNode,
        kl_divergence_cache::KLDCache, pr_constraint_cache::PRCache)
    @assert variables(pc_node1) == variables(pc_node2) "Both nodes not normalized for same vtree node"

    if (pc_node1, pc_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[pc_node1, pc_node2]
    else
        kl_divergence(children(pc_node1)[1], pc_node2, kl_divergence_cache, pr_constraint_cache)
        kl_divergence(children(pc_node1)[2], pc_node2, kl_divergence_cache, pr_constraint_cache)
        if literal(children(pc_node1)[1]) == literal(pc_node2)
            return get!(kl_divergence_cache, (pc_node1, pc_node2),
                pc_node1.log_probs[1] * exp(pc_node1.log_probs[1])
            )
        else
            return get!(kl_divergence_cache, (pc_node1, pc_node2),
                pc_node1.log_probs[2] * exp(pc_node1.log_probs[2])
            )
        end
    end
end

function kl_divergence(pc_node1::StructProbLiteralNode, pc_node2::StructSumNode,
        kl_divergence_cache::KLDCache, pr_constraint_cache::PRCache)
    @assert variables(pc_node1) == variables(pc_node2) "Both nodes not normalized for same vtree node"

    if (pc_node1, pc_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[pc_node1, pc_node2]
    else
        kl_divergence(pc_node1, children(pc_node2)[1], kl_divergence_cache, pr_constraint_cache)
        kl_divergence(pc_node1, children(pc_node2)[2], kl_divergence_cache, pr_constraint_cache)
        if literal(pc_node1) == literal(children(pc_node2)[1])
            return get!(kl_divergence_cache, (pc_node1, pc_node2),
                -pc_node2.log_probs[1]
            )
        else
            return get!(kl_divergence_cache, (pc_node1, pc_node2),
                -pc_node2.log_probs[2]
            )
        end
    end
end
