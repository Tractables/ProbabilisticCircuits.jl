export MAR, marginal_log_likelihood_per_instance, 
MPE, MAP, sample

"""
Marginal queries
"""
function marginal_log_likelihood_per_instance(pc::ProbCircuit, data)
    evaluate_exp(pc, data)
end
MAR = marginal_log_likelihood_per_instance


"""
Most Probable Explanation (MPE), aka MAP
"""
@inline function MAP(pc::ProbCircuit, evidence)::BitMatrix
    MPE(pc, evidence)
end

function MPE(pc::ProbCircuit, evidence)::BitMatrix
    mlls = marginal_log_likelihood_per_instance(pc, evidence)
    
    ans = falses(num_examples(evidence), num_features(evidence))
    active_samples = trues(num_examples(evidence))

    function mpe_simulate(node::Union{PlainProbLiteralNode, StructProbLiteralNode}, active_samples::BitVector, result::BitMatrix)
        if ispositive(node)
            result[active_samples, variable(node)] .= 1
        else
            result[active_samples, variable(node)] .= 0
        end
    end
    
    function mpe_simulate(node::Union{PlainSumNode, StructSumNode}, active_samples::BitVector, result::BitMatrix)
        prs = zeros(length(children(node)), size(active_samples)[1] )
        @simd  for i=1:length(children(node))
            prs[i,:] .= get_exp_upflow(children(node)[i]) .+ (node.log_probs[i])
        end
    
        max_child_ids = [a[1] for a in argmax(prs, dims = 1) ]
        @simd for i=1:length(children(node))
            # Only active for this child if it was the max for that sample
            ids = convert(BitVector, active_samples .* (max_child_ids .== i)[1,:])
            mpe_simulate(children(node)[i], ids, result)
        end
    end
    
    function mpe_simulate(node::Union{PlainMulNode, StructMulNode}, active_samples::BitVector, result::BitMatrix)
        for child in children(node)
            mpe_simulate(child, active_samples, result)
        end
    end

    mpe_simulate(pc, active_samples, ans)
    ans
end


##################
# Sampling from a psdd
##################

"""
Sample from a PSDD without any evidence
"""
function sample(circuit::ProbCircuit)::AbstractVector{Bool}

    simulate(node::Union{PlainProbLiteralNode, StructProbLiteralNode}) = begin
        inst[variable(node)] = ispositive(node) ? 1 : 0
    end
    
    simulate(node::Union{PlainSumNode, StructSumNode}) = begin
        idx = sample(exp.(node.log_probs))
        simulate(children(node)[idx])
    end

    simulate(node::Union{PlainMulNode, StructMulNode}) = foreach(simulate, children(node))

    inst = Dict{Var,Int64}()
    simulate(circuit)
    len = length(keys(inst))
    ans = Vector{Bool}()
    for i = 1:len
        push!(ans, inst[i])
    end
    ans
end


"""
Sampling with Evidence from a psdd.
"""
function sample(circuit::ProbCircuit, evidence)::AbstractVector{Bool}

    @assert num_examples(evidence) == 1 "evidence have to be one example"
    
    simulate(node::Union{PlainProbLiteralNode, StructProbLiteralNode}) = begin
        inst[variable(node)] = ispositive(node) ? 1 : 0
    end
    
    function simulate(node::Union{PlainSumNode, StructSumNode})
        prs = [get_exp_upflow(ch)[1] for ch in children(node)] # #evidence == 1
        idx = sample(exp.(node.log_probs .+ prs))
        simulate(children(node)[idx])
    end
    
    simulate(node::Union{PlainMulNode, StructMulNode}) = foreach(simulate, children(node))

    evaluate_exp(circuit, evidence)

    inst = Dict{Var,Int64}()
    simulate(circuit)
    len = length(keys(inst))
    ans = Vector{Bool}()
    for i = 1:len
        push!(ans, inst[i])
    end
    ans
end


"""
Uniformly sample based on the probability of the items and return the selected index
"""
function sample(probs::AbstractVector{<:Number})::Int32
    z = sum(probs)
    q = rand() * z
    cur = 0.0
    for i = 1:length(probs)
        cur += probs[i]
        if q <= cur
            return i
        end
    end
    return length(probs)
end
