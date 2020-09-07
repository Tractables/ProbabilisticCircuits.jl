export ParamBitCircuit

"A `BitCircuit` with parameters attached to the elements"
struct ParamBitCircuit{V,M,W}
    bitcircuit::BitCircuit{V,M}
    params::W
end

function ParamBitCircuit(pc::ProbCircuit, data)
    logprobs::Vector{Float64} = Vector{Float64}()
    on_decision(n, cs, layer_id, decision_id, first_element, last_element) = begin
        if isnothing(n) # this decision node is not part of the PC
            # @assert first_element == last_element
            push!(logprobs, 0.0)
        else
            # @assert last_element-first_element+1 == length(n.log_probs) 
            append!(logprobs, n.log_probs)
        end
    end
    bc = BitCircuit(pc, data; on_decision)
    ParamBitCircuit(bc, logprobs)
end

function ParamBitCircuit(lc::LogisticCircuit, nc, data)
    thetas::Vector{Vector{Float64}} = Vector{Vector{Float64}}()
    on_decision(n, cs, layer_id, decision_id, first_element, last_element) = begin
        if isnothing(n)
            # @assert first_element == last_element
            push!(thetas, zeros(Float64, nc))
        else
            # @assert last_element-first_element+1 == length(n.log_probs) "$last_element-$first_element+1 != $(length(n.log_probs))"
            for theta in eachrow(n.thetas)
                push!(thetas, theta)
            end
        end
    end
    bc = BitCircuit(lc, data; on_decision)
    ParamBitCircuit(bc, permutedims(hcat(thetas...), (2, 1)))
end



#######################
## Helper functions ###
#######################

import LogicCircuits: num_nodes, num_elements, num_features

num_nodes(c::ParamBitCircuit) = num_nodes(c.bitcircuit)
num_elements(c::ParamBitCircuit) = num_elements(c.bitcircuit)
num_features(c::ParamBitCircuit) = num_features(c.bitcircuit)

import LogicCircuits: to_gpu, to_cpu, isgpu #extend

to_gpu(c::ParamBitCircuit) = 
    ParamBitCircuit(to_gpu(c.bitcircuit), to_gpu(c.params))

to_cpu(c::ParamBitCircuit) = 
    ParamBitCircuit(to_cpu(c.bitcircuit), to_cpu(c.params))


isgpu(c::ParamBitCircuit) = 
    isgpu(c.bitcircuit) && isgpu(c.params)
