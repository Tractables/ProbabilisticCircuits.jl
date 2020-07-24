export pr_constraint, kl_divergence, entropy

const StrutCircuit = Union{ProbCircuit, StructLogicCircuit}
const KLDCache = Dict{Tuple{ProbCircuit, ProbCircuit}, Float64}
const PRCache = Dict{Tuple{ProbCircuit, StrutCircuit}, Float64}

# Arthur Choi, Guy Van den Broeck, and Adnan Darwiche. Tractable learning for structured probability
# spaces: A case study in learning preference distributions. In Proceedings of IJCAI, 2015.

"""
Calculate the probability of the logic formula given by sdd for the psdd
"""
function pr_constraint(psdd_node::ProbCircuit, sdd_node::StrutCircuit,
    cache::PRCache=PRCache())::Float64
    
    # Cache hit
    if (psdd_node, sdd_node) in keys(cache) 
        return cache[psdd_node, sdd_node]
    
    # Boundary cases
    elseif psdd_node isa ProbLiteralNode
        # Both are literals, just check whether they agrees with each other 
        if isliteralgate(sdd_node)
            if literal(psdd_node) == literal(sdd_node)
                return get!(cache, (psdd_node, sdd_node), 1.0)
            else
                return get!(cache, (psdd_node, sdd_node), 0.0)
            end
        else
            pr_constraint(psdd_node, children(sdd_node)[1], cache)
            if length(children(sdd_node)) > 1
                pr_constraint(psdd_node, children(sdd_node)[2], cache)
                return get!(cache, (psdd_node, sdd_node), 1.0)
            else
                return get!(cache, (psdd_node, sdd_node),
                    literal(children(sdd_node)[1]) == literal(psdd_node) ? 1.0 : 0.0)
            end
        end
    
    # The psdd is true
    elseif children(psdd_node)[1] isa ProbLiteralNode 
        theta = exp(psdd_node.log_thetas[1])
        return get!(cache, (psdd_node, sdd_node),
            theta * pr_constraint(children(psdd_node)[1], sdd_node, cache) +
            (1.0 - theta) * pr_constraint(children(psdd_node)[2], sdd_node, cache))
    
    # Both psdds are not trivial
    else 
        prob = 0.0
        for (prob⋀_node, log_theta) in zip(children(psdd_node), psdd_node.log_thetas)
            p = children(prob⋀_node)[1]
            s = children(prob⋀_node)[2]

            theta = exp(log_theta)
            for sdd⋀_node in children(sdd_node)
                r = children(sdd⋀_node)[1]
                t = children(sdd⋀_node)[2]
                prob += theta * pr_constraint(p, r, cache) * pr_constraint(s, t, cache)
            end
        end
        return get!(cache, (psdd_node, sdd_node), prob)
    end
end


""""
Calculate entropy of the distribution of the input psdd."
"""
function entropy(psdd_node::Prob⋁Node, psdd_entropy_cache::Dict{ProbCircuit, Float64}=Dict{ProbCircuit, Float64}())::Float64
    if psdd_node in keys(psdd_entropy_cache)
        return psdd_entropy_cache[psdd_node]
    elseif children(psdd_node)[1] isa ProbLiteralNode
        return get!(psdd_entropy_cache, psdd_node,
            - exp(psdd_node.log_thetas[1]) * psdd_node.log_thetas[1] -
            exp(psdd_node.log_thetas[2]) * psdd_node.log_thetas[2])
    else
        local_entropy = 0.0
        for (prob⋀_node, log_prob) in zip(children(psdd_node), psdd_node.log_thetas)
            p = children(prob⋀_node)[1]
            s = children(prob⋀_node)[2]

            local_entropy += exp(log_prob) * (entropy(p, psdd_entropy_cache) +
                entropy(s, psdd_entropy_cache) - log_prob)
        end
        return get!(psdd_entropy_cache, psdd_node, local_entropy)
    end
end

function entropy(psdd_node::Prob⋀Node, psdd_entropy_cache::Dict{ProbCircuit, Float64})::Float64
    return get!(psdd_entropy_cache, children(psdd_node)[1], entropy(children(psdd_node)[1], psdd_entropy_cache)) +
        get!(psdd_entropy_cache, children(psdd_node)[2], entropy(children(psdd_node)[2], psdd_entropy_cache))
end

function entropy(psdd_node::ProbLiteralNode, psdd_entropy_cache::Dict{ProbCircuit, Float64})::Float64
    return get!(psdd_entropy_cache, psdd_node, 0.0)
end

"Calculate KL divergence calculation for psdds that are not necessarily identical"
function kl_divergence(psdd_node1::Prob⋁Node, psdd_node2::Prob⋁Node,
        kl_divergence_cache::KLDCache=KLDCache(), pr_constraint_cache::PRCache=PRCache())
    @assert !(psdd_node1 isa Prob⋀Node || psdd_node2 isa Prob⋀Node) "Prob⋀ not a valid PSDD node for KL-Divergence"

    # Check if both nodes are normalized for same vtree node
    @assert variables(psdd_node1.origin.vtree) == variables(psdd_node2.origin.vtree) "Both nodes not normalized for same vtree node"

    if (psdd_node1, psdd_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[(psdd_node1, psdd_node2)]
    elseif children(psdd_node1)[1] isa ProbLiteralNode
        if psdd_node2 isa ProbLiteralNode
            kl_divergence(children(psdd_node1)[1], psdd_node2, kl_divergence_cache, pr_constraint_cache)
            kl_divergence(children(psdd_node1)[2], psdd_node2, kl_divergence_cache, pr_constraint_cache)
            if literal(children(psdd_node1)[1]) == literal(psdd_node2)
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
            kl_divergence(children(psdd_node1)[1], children(psdd_node2)[1], kl_divergence_cache, pr_constraint_cache)
            kl_divergence(children(psdd_node1)[1], children(psdd_node2)[2], kl_divergence_cache, pr_constraint_cache)
            kl_divergence(children(psdd_node1)[2], children(psdd_node2)[1], kl_divergence_cache, pr_constraint_cache)
            kl_divergence(children(psdd_node1)[2], children(psdd_node2)[2], kl_divergence_cache, pr_constraint_cache)
            # There are two possible matches
            if literal(children(psdd_node1)[1]) == literal(children(psdd_node2)[1])
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
        kld = 0.0

        # loop through every combination of prim and sub
        for (prob⋀_node1, log_theta1) in zip(children(psdd_node1), psdd_node1.log_thetas)
            for (prob⋀_node2, log_theta2) in zip(children(psdd_node2), psdd_node2.log_thetas)
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

function kl_divergence(psdd_node1::ProbLiteralNode, psdd_node2::ProbLiteralNode,
        kl_divergence_cache::KLDCache, pr_constraint_cache::PRCache)
    # Check if literals are over same variables in vtree
   @assert variables(psdd_node1.origin.vtree) == variables(psdd_node2.origin.vtree) "Both nodes not normalized for same vtree node"

    if (psdd_node1, psdd_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[psdd_node1, psdd_node2]
    else
        # In this case probability is 1, kl divergence is 0
        return get!(kl_divergence_cache, (psdd_node1, psdd_node2), 0.0)
    end
end

function kl_divergence(psdd_node1::Prob⋁Node, psdd_node2::ProbLiteralNode,
        kl_divergence_cache::KLDCache, pr_constraint_cache::PRCache)
    @assert variables(psdd_node1.origin.vtree) == variables(psdd_node2.origin.vtree) "Both nodes not normalized for same vtree node"

    if (psdd_node1, psdd_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[psdd_node1, psdd_node2]
    else
        kl_divergence(children(psdd_node1)[1], psdd_node2, kl_divergence_cache, pr_constraint_cache)
        kl_divergence(children(psdd_node1)[2], psdd_node2, kl_divergence_cache, pr_constraint_cache)
        if literal(children(psdd_node1)[1]) == literal(psdd_node2)
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

function kl_divergence(psdd_node1::ProbLiteralNode, psdd_node2::Prob⋁Node,
        kl_divergence_cache::KLDCache, pr_constraint_cache::PRCache)
    @assert variables(psdd_node1.origin.vtree) == variables(psdd_node2.origin.vtree) "Both nodes not normalized for same vtree node"

    if (psdd_node1, psdd_node2) in keys(kl_divergence_cache) # Cache hit
        return kl_divergence_cache[psdd_node1, psdd_node2]
    else
        kl_divergence(psdd_node1, children(psdd_node2)[1], kl_divergence_cache, pr_constraint_cache)
        kl_divergence(psdd_node1, children(psdd_node2)[2], kl_divergence_cache, pr_constraint_cache)
        if literal(psdd_node1) == literal(children(psdd_node2)[1])
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
