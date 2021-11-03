export ParamBitCircuit

"A `BitCircuit` with parameters attached to the elements"
mutable struct ParamBitCircuit{V,M,W}
    bitcircuit::BitCircuit{V,M}
    params::W
end

function ParamBitCircuit(pc::ProbCircuit, data)
    logprobs::Vector{Float64} = Vector{Float64}()
    sizehint!(logprobs, num_edges(pc))
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

function ParamBitCircuit(spc::SharedProbCircuit, data; component_idx)
    logprobs::Vector{Float64} = Vector{Float64}()
    sizehint!(logprobs, num_edges(spc))
    on_decision(n, cs, layer_id, decision_id, first_element, last_element) = begin
        if isnothing(n) # this decision node is not part of the PC
            # @assert first_element == last_element
            push!(logprobs, 0.0)
        else
            # @assert last_element-first_element+1 == length(n.log_probs) 
            append!(logprobs, n.log_probs[:, component_idx])
        end
    end
    bc = BitCircuit(spc, data; on_decision)
    ParamBitCircuit(bc, logprobs)
end

#######################
## Helper functions ###
#######################

params(c::ParamBitCircuit) = c.params

import LogicCircuits: num_nodes, num_elements, num_features, num_leafs, nodes, elements

num_nodes(c::ParamBitCircuit) = num_nodes(c.bitcircuit)
num_elements(c::ParamBitCircuit) = num_elements(c.bitcircuit)
num_features(c::ParamBitCircuit) = num_features(c.bitcircuit)
num_leafs(c::ParamBitCircuit) = num_leafs(c.bitcircuit)

nodes(c::ParamBitCircuit) = nodes(c.bitcircuit)
elements(c::ParamBitCircuit) = elements(c.bitcircuit)

import LogicCircuits: to_gpu, to_cpu, isgpu #extend

to_gpu(c::ParamBitCircuit) = 
    ParamBitCircuit(to_gpu(c.bitcircuit), to_gpu(c.params))

to_cpu(c::ParamBitCircuit) = 
    ParamBitCircuit(to_cpu(c.bitcircuit), to_cpu(c.params))


isgpu(c::ParamBitCircuit) = 
    isgpu(c.bitcircuit) && isgpu(c.params)
