using CUDA
using LogicCircuits

export LayeredParameterCircuit, CuLayeredParameterCircuit
export class_likelihood_and_flow, class_weights_and_flow

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

function class_likelihood_and_flow(circuit::CuLayeredParameterCircuit, nc::Integer, data::CuMatrix{Float32}, reuse_up=nothing, reuse_down=nothing, reuse_cw=nothing)
    cw, flow = class_weights_and_flow(circuit, nc, data, reuse_up, reuse_down, reuse_cw)
    return @. 1.0 / (1.0 + exp(-cw)), flow
end

function class_weights_and_flow(circuit::CuLayeredParameterCircuit, nc::Integer, data::CuMatrix{Float32}, reuse_up=nothing, reuse_down=nothing, reuse_cw=nothing)
    _, edge_flow, _ = compute_flows2(circuit.layered_circuit, data, reuse_up, reuse_down)
    cw = calculate_class_weights(circuit, nc, data, edge_flow, reuse_cw)
    return cw, edge_flow
end

function calculate_class_weights(circuit::CuLayeredParameterCircuit, nc::Integer, data::CuMatrix{Float32}, flow, reuse_cw=nothing)
    ne = num_examples(data)
    cw = if reuse_cw isa CuMatrix{Float32} && size(reuse_cw) == (ne, nc)
        reuse_cw .= zero(Float32)
        reuse_cw
    else
        CUDA.zeros(Float32, ne, nc)
    end

    dec_per_thread = 8
    CUDA.@sync for i = 1:length(circuit.layered_circuit.layers)
        circuit_layer = circuit.layered_circuit.layers[i]
        parameter_layer = circuit.layered_parameters[i]
        ndl = num_decisions(circuit_layer)
        num_threads = balance_threads(ne, ndl / dec_per_thread, 8)
        num_blocks = (ceil(Int, ne / num_threads[1]), ceil(Int, ndl / num_threads[2] / dec_per_thread)) 
        @cuda threads=num_threads blocks=num_blocks calculate_class_weights_layer_kernel_cuda(cw, flow, circuit_layer.decisions, parameter_layer)
    end
    
    return cw
end

function calculate_class_weights_layer_kernel_cuda(cw, flow, decisions, parameters)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().y
    ne, nc = size(cw)
    _, num_decisions = size(decisions)
    
    for j = index_x:stride_x:ne
        for i = index_y:stride_y:num_decisions
            first_elem = @inbounds decisions[2, i]
            last_elem = @inbounds decisions[3, i]
            for e = first_elem:last_elem
                # following needs to be memory safe
                for class=1:nc
                    CUDA.@atomic cw[j, class] += flow[j, e] * parameters[class, e] #atomic is automatically inbounds
                end
            end
        end
    end
    
    return nothing
end



# TODO; parameter learning