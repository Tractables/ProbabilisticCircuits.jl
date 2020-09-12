export sample

##################
# Sampling from a psdd
##################

"""
Sample from a PC without any evidence
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
