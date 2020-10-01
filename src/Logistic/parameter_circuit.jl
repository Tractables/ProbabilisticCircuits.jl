using CUDA
using LogicCircuits

export LayeredParameterCircuit, CuLayeredParameterCircuit
export class_likelihood, class_weights
export one_hot, learn_parameters, update_parameters

#############################################################
############## This is the old implementation ###############
#### Not intended to be used under the current framework ####
#############################################################


# in a parameter circuit
# 1 is true, 2 is false
const TRUE_ID = Int32(1)
const FALSE_ID = Int32(2)

struct LayeredParameterCircuit
    layered_circuit::LayeredBitCircuit
    layered_parameters::Vector{Matrix{Float32}}
end

LayeredParameterCircuit(circuit::LogisticCircuit, nc::Integer, num_features::Integer) = begin
    @assert is⋁gate(circuit)
    decisions::Vector{Vector{Int32}} = Vector{Vector{Int32}}()
    elements::Vector{Vector{Int32}} = Vector{Vector{Int32}}()
    parameters::Vector{Vector{Float32}} = Vector{Vector{Float32}}()
    num_decisions::Int32 = 2 * num_features + 2
    num_elements::Vector{Int32} = Vector{Int32}()
    # num_parameters always equals num_elements

    ensure_layer(i) = begin
        if length(decisions) < i
            # add a new layer
            push!(decisions, Int32[])
            push!(elements, Int32[])
            push!(parameters, Float32[])
            push!(num_elements, 0)
        end
    end
    
    f_con(n) = LayeredDecisionId(0, istrue(n) ? TRUE_ID : FALSE_ID)
    f_lit(n) = LayeredDecisionId(0, 
        ispositive(n) ? Int32(2 + variable(n)) : Int32(2 + num_features + variable(n)))

    f_and(n, cs) = begin
        @assert length(cs) == 2
        LayeredDecisionId[cs[1], cs[2]]
    end
    f_or(n, cs) = begin
        num_decisions += 1
        # determine layer
        layer_id = zero(Int32)
        for c in cs
            if c isa Vector{LayeredDecisionId}
                @assert length(c) == 2
                layer_id = max(layer_id, c[1].layer_id, c[2].layer_id)
            else
                @assert c isa LayeredDecisionId
                layer_id = max(layer_id, c.layer_id)
            end
        end
        layer_id += 1
        ensure_layer(layer_id)
        first_element = num_elements[layer_id] + 1
        foreach(cs, eachrow(n.thetas)) do c, theta
            @assert size(theta)[1] == nc
            append!(parameters[layer_id], theta)
            num_elements[layer_id] += 1
            if c isa Vector{LayeredDecisionId}
                push!(elements[layer_id], c[1].decision_id, c[2].decision_id)
            else
                push!(elements[layer_id], c.decision_id, TRUE_ID)
            end
        end
        push!(decisions[layer_id], num_decisions, first_element, num_elements[layer_id])
        LayeredDecisionId(layer_id, num_decisions)
    end

    foldup_aggregate(circuit, f_con, f_lit, f_and, f_or, 
        Union{LayeredDecisionId,Vector{LayeredDecisionId}})
    
    circuit_layers = map(decisions, elements) do d, e
        Layer(reshape(d, 3, :), reshape(e, 2, :))
    end
    parameter_layers = map(parameters) do p
        reshape(p, nc, :)
    end
    return LayeredParameterCircuit(LayeredBitCircuit(circuit_layers), parameter_layers)
end

struct CuLayeredParameterCircuit
    layered_circuit::CuLayeredBitCircuit
    layered_parameters::Vector{CuMatrix{Float32}}
    CuLayeredParameterCircuit(l::LayeredParameterCircuit) = new(CuLayeredBitCircuit(l.layered_circuit), map(CuMatrix, l.layered_parameters))
end



function class_likelihood(circuit::CuLayeredParameterCircuit, nc::Integer, data::CuMatrix{Float32}, reuse_up=nothing, reuse_down=nothing, reuse_cp=nothing)
    cw, flow, v = class_weights(circuit, nc, data, reuse_up, reuse_down, reuse_cp)
    one = Float32(1.0)
    return @. one / (one + exp(-cw)), flow, v
end

function class_weights(circuit::CuLayeredParameterCircuit, nc::Integer, data::CuMatrix{Float32}, reuse_up=nothing, reuse_down=nothing, reuse_cw=nothing)
    flow, v = compute_flows2(circuit.layered_circuit, data, reuse_up, reuse_down)
    cw = calculate_class_weights(circuit, nc, data, v, flow, reuse_cw)
    return cw, flow, v
end

function calculate_class_weights(circuit::CuLayeredParameterCircuit, nc::Integer, data::CuMatrix{Float32}, v, flow, reuse_cw=nothing)
    ne = num_examples(data)
    cw = if reuse_cw isa CuMatrix{Float32} && size(reuse_cw) == (ne, nc)
        reuse_cw .= zero(Float32)
        reuse_cw
    else
        CUDA.zeros(Float32, ne, nc)
    end

    dec_per_thread = 4
    CUDA.@sync for i = 1:length(circuit.layered_circuit.layers)
        circuit_layer = circuit.layered_circuit.layers[i]
        parameter_layer = circuit.layered_parameters[i]
        ndl = num_decisions(circuit_layer)
        num_threads = balance_threads(ne, ndl / dec_per_thread, 8)
        num_blocks = ceil(Int, ne / num_threads[1]), ceil(Int, ndl / num_threads[2] / dec_per_thread)
        @cuda threads=num_threads blocks=num_blocks calculate_class_weights_layer_kernel_cuda(cw, v, flow, circuit_layer.decisions, circuit_layer.elements, parameter_layer)
    end
    
    return cw
end

function calculate_class_weights_layer_kernel_cuda(cw, v, flow, decisions, elements, parameters)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().y
    ne, nc = size(cw)
    _, num_decisions = size(decisions)
    
    for j = index_x:stride_x:ne
        for i = index_y:stride_y:num_decisions
            decision_id = @inbounds decisions[1, i]
            n_up = @inbounds v[j, decision_id]
            if n_up > zero(Float32)
                first_elem = @inbounds decisions[2, i]
                last_elem = @inbounds decisions[3, i]
                n_down = @inbounds flow[j, decision_id]
                for e = first_elem:last_elem
                    e1 = @inbounds elements[1, first_elem]
                    e2 = @inbounds elements[2, first_elem]
                    e_up = @inbounds (v[j, e1] * v[j, e2])
                    edge_flow = e_up / n_up * n_down
                    # following needs to be memory safe
                    for class=1:nc
                        @CUDA.atomic cw[j, class] += edge_flow * parameters[class, e] # atomic is automatically inbounds
                    end
                end
            end
        end
    end
    
    return nothing
end



function one_hot(labels::Vector, nc::Integer)    
    ne = length(labels) 
    one_hot_labels = zeros(Float32, ne, nc)
    for (i, j) in enumerate(labels)
        one_hot_labels[i, j + 1] = 1.0
    end
    one_hot_labels
end

function learn_parameters(circuit::CuLayeredParameterCircuit, nc::Integer, data::CuMatrix{Float32}, labels::CuMatrix{Float32}, reuse_up=nothing, reuse_down=nothing, reuse_cp=nothing, num_epochs=20, step_size=0.0001)
    cp, flow, v = class_likelihood(circuit, nc, data, reuse_up, reuse_down, reuse_cp)
    update_parameters(circuit, labels, cp, flow, step_size)
    for _ = 2:num_epochs
        cp, flow, v = class_likelihood(circuit, nc, data, v, flow, cp)
        update_parameters(circuit, labels, cp, v, flow, step_size)
    end
    return nothing
end

function update_parameters(circuit::CuLayeredParameterCircuit, labels, cp, v, flow, step_size=0.0001)
    _, nc = size(labels)
    step_size = Float32(step_size)
    CUDA.@sync for i = 1:length(circuit.layered_circuit.layers)
        circuit_layer = circuit.layered_circuit.layers[i]
        flow_layer = flow[i]
        parameter_layer = circuit.layered_parameters[i]
        ndl = num_decisions(circuit_layer)
        num_threads = balance_threads(ndl, nc, 6)
        num_threads = num_threads[1], num_threads[2], 
        num_blocks = ceil(Int, ndl / num_threads[1]), ceil(Int, nc / num_threads[2]), 4
        @cuda threads=num_threads blocks=num_blocks update_parameters_layer_kernel_cuda(labels, cp, flow_layer, circuit_layer.decisions, parameter_layer, step_size)
    end
    return nothing
end

function update_parameters_layer_kernel_cuda(labels, cp, flow, decisions, parameters, step_size)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    index_z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().y
    stride_z = blockDim().z * gridDim().z
    ne, nc = size(labels)
    _, num_decisions = size(decisions)
    
    for class = index_y:stride_y:nc
        for i = index_x:stride_x:num_decisions
            first_elem = @inbounds decisions[2, i]
            last_elem = @inbounds decisions[3, i]
            for e = first_elem:last_elem
                for j = index_z:stride_z:ne
                    edge_flow = e_up / n_up * n_down
                    u = @inbounds edge_flow * (cp[j, class] - labels[j, class]) * step_size
                    # following needs to be memory safe
                    @inbounds parameters[class, e] -= u 
                end
            end
        end
    end
    
    return nothing
end