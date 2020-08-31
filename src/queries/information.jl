export kl_divergence

const KLDCache = Dict{Tuple{ProbCircuit, ProbCircuit}, Float64}

""""
Calculate entropy of the distribution of the input psdd."
"""

import ..Utils: entropy
function entropy(psdd_node::StructSumNode, psdd_entropy_cache::Dict{ProbCircuit, Float64}=Dict{ProbCircuit, Float64}())::Float64
    if psdd_node in keys(psdd_entropy_cache)
        return psdd_entropy_cache[psdd_node]
    elseif children(psdd_node)[1] isa StructProbLiteralNode
        return get!(psdd_entropy_cache, psdd_node,
            - exp(psdd_node.log_probs[1]) * psdd_node.log_probs[1] -
            exp(psdd_node.log_probs[2]) * psdd_node.log_probs[2])
    else
        local_entropy = 0.0
        for (prob⋀_node, log_prob) in zip(children(psdd_node), psdd_node.log_probs)
            p = children(prob⋀_node)[1]
            s = children(prob⋀_node)[2]

            local_entropy += exp(log_prob) * (entropy(p, psdd_entropy_cache) +
                entropy(s, psdd_entropy_cache) - log_prob)
        end
        return get!(psdd_entropy_cache, psdd_node, local_entropy)
    end
end

function entropy(psdd_node::StructMulNode, psdd_entropy_cache::Dict{ProbCircuit, Float64})::Float64
    return get!(psdd_entropy_cache, children(psdd_node)[1], entropy(children(psdd_node)[1], psdd_entropy_cache)) +
        get!(psdd_entropy_cache, children(psdd_node)[2], entropy(children(psdd_node)[2], psdd_entropy_cache))
end

function entropy(psdd_node::StructProbLiteralNode, psdd_entropy_cache::Dict{ProbCircuit, Float64})::Float64
    return get!(psdd_entropy_cache, psdd_node, 0.0)
end

"Calculate KL divergence calculation for psdds that are not necessarily identical"
function kl_divergence(psdd_node1::StructSumNode, psdd_node2::StructSumNode,
        kl_divergence_cache::KLDCache=KLDCache(), pr_constraint_cache::PRCache=PRCache())
    @assert !(psdd_node1 isa StructMulNode || psdd_node2 isa StructMulNode) "Prob⋀ not a valid PSDD node for KL-Divergence"

    # Check if both nodes are normalized for same vtree node
    @assert variables(psdd_node1) == variables(psdd_node2) "Both nodes not normalized for same vtree node"

    if (psdd_node1, psdd_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[(psdd_node1, psdd_node2)]
    elseif children(psdd_node1)[1] isa StructProbLiteralNode
        if psdd_node2 isa StructProbLiteralNode
            kl_divergence(children(psdd_node1)[1], psdd_node2, kl_divergence_cache, pr_constraint_cache)
            kl_divergence(children(psdd_node1)[2], psdd_node2, kl_divergence_cache, pr_constraint_cache)
            if literal(children(psdd_node1)[1]) == literal(psdd_node2)
                return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                    psdd_node1.log_probs[1] * exp(psdd_node1.log_probs[1])
                )
            else
                return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                    psdd_node1.log_probs[2] * exp(psdd_node1.log_probs[2])
                )
            end
        else
            # The below four lines actually assign zero, but still we need to
            # call it.
            kl_divergence(children(psdd_node1)[1], children(psdd_node2)[1], kl_divergence_cache, pr_constraint_cache)
            kl_divergence(children(psdd_node1)[1], children(psdd_node2)[2], kl_divergence_cache, pr_constraint_cache)
            kl_divergence(children(psdd_node1)[2], children(psdd_node2)[1], kl_divergence_cache, pr_constraint_cache)
            kl_divergence(children(psdd_node1)[2], children(psdd_node2)[2], kl_divergence_cache, pr_constraint_cache)
            # There are two possible matches
            if literal(children(psdd_node1)[1]) == literal(children(psdd_node2)[1])
                return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                    exp(psdd_node1.log_probs[1]) * (psdd_node1.log_probs[1] - psdd_node2.log_probs[1]) +
                    exp(psdd_node1.log_probs[2]) * (psdd_node1.log_probs[2] - psdd_node2.log_probs[2])
                )
            else
                return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                    exp(psdd_node1.log_probs[1]) * (psdd_node1.log_probs[1] - psdd_node2.log_probs[2]) +
                    exp(psdd_node1.log_probs[2]) * (psdd_node1.log_probs[2] - psdd_node2.log_probs[1])
                )
            end
        end
    else # the normal case
        kld = 0.0

        # loop through every combination of prim and sub
        for (prob⋀_node1, log_theta1) in zip(children(psdd_node1), psdd_node1.log_probs)
            for (prob⋀_node2, log_theta2) in zip(children(psdd_node2), psdd_node2.log_probs)
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
        return get!(kl_divergence_cache, (psdd_node1, psdd_node2), kld)
    end
end

function kl_divergence(psdd_node1::StructProbLiteralNode, psdd_node2::StructProbLiteralNode,
        kl_divergence_cache::KLDCache, pr_constraint_cache::PRCache)
    # Check if literals are over same variables in vtree
   @assert variables(psdd_node1) == variables(psdd_node2) "Both nodes not normalized for same vtree node"

    if (psdd_node1, psdd_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[psdd_node1, psdd_node2]
    else
        # In this case probability is 1, kl divergence is 0
        return get!(kl_divergence_cache, (psdd_node1, psdd_node2), 0.0)
    end
end

function kl_divergence(psdd_node1::StructSumNode, psdd_node2::StructProbLiteralNode,
        kl_divergence_cache::KLDCache, pr_constraint_cache::PRCache)
    @assert variables(psdd_node1) == variables(psdd_node2) "Both nodes not normalized for same vtree node"

    if (psdd_node1, psdd_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[psdd_node1, psdd_node2]
    else
        kl_divergence(children(psdd_node1)[1], psdd_node2, kl_divergence_cache, pr_constraint_cache)
        kl_divergence(children(psdd_node1)[2], psdd_node2, kl_divergence_cache, pr_constraint_cache)
        if literal(children(psdd_node1)[1]) == literal(psdd_node2)
            return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                psdd_node1.log_probs[1] * exp(psdd_node1.log_probs[1])
            )
        else
            return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                psdd_node1.log_probs[2] * exp(psdd_node1.log_probs[2])
            )
        end
    end
end

function kl_divergence(psdd_node1::StructProbLiteralNode, psdd_node2::StructSumNode,
        kl_divergence_cache::KLDCache, pr_constraint_cache::PRCache)
    @assert variables(psdd_node1) == variables(psdd_node2) "Both nodes not normalized for same vtree node"

    if (psdd_node1, psdd_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[psdd_node1, psdd_node2]
    else
        kl_divergence(psdd_node1, children(psdd_node2)[1], kl_divergence_cache, pr_constraint_cache)
        kl_divergence(psdd_node1, children(psdd_node2)[2], kl_divergence_cache, pr_constraint_cache)
        if literal(psdd_node1) == literal(children(psdd_node2)[1])
            return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                -psdd_node2.log_probs[1]
            )
        else
            return get!(kl_divergence_cache, (psdd_node1, psdd_node2),
                -psdd_node2.log_probs[2]
            )
        end
    end
end
