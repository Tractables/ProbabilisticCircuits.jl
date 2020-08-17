using CUDA
using LogicCircuits

export ParameterCircuit, CuParameterCircuit
export class_likelihood_and_flow, class_weights_and_flow

export LayeredParameterCircuit, CuLayeredParameterCircuit
export class_likelihood_and_flow2, class_weights_and_flow2


# In a parameter circuit,
# 1 is true, 2 is false
const TRUE_BITS = Int32(1)
const FALSE_BITS = Int32(2)
# 3:nf+2 are nf positive literals
# nf+3:2nf+2 are nf negative literals
# 2nf+2:end are inner decision nodes


struct ParameterCircuit
    decisions::Matrix{Int32}
    elements::Matrix{Int32}
    parameters::Matrix{Float32}
end

ParameterCircuit(circuit::LogisticCircuit, nc::Integer, num_features::Integer) = begin
    @assert is⋁gate(circuit)
    decisions::Vector{Int32} = Vector{Int32}()
    elements::Vector{Int32} = Vector{Int32}()
    parameters::Vector{Float32} = Vector{Float32}()
    num_decisions::Int32 = 2 * num_features + 2
    num_elements::Int32 = 0
    # num_parameters always equals num_elements
    
    f_con(n) = istrue(n) ? TRUE_BITS : FALSE_BITS
    f_lit(n) = ispositive(n) ? Int32(2 + variable(n)) : Int32(2 + num_features + variable(n))
    f_and(n, cs) = begin
        @assert length(cs) == 2
        Int32[cs[1], cs[2]]
    end
    f_or(n, cs) = begin
        first_element = num_elements + 1
        foreach(cs, eachrow(n.thetas)) do c, theta
            @assert size(theta)[1] == nc
            append!(parameters, theta)
            num_elements += 1
            if c isa Vector{Int32}
                @assert length(c) == 2
                push!(elements, c[1], c[2])
            else
                @assert c isa Int32
                push!(elements, c, TRUE_BITS)
            end
        end
        num_decisions += 1
        push!(decisions, first_element, num_elements)
        num_decisions
    end
    foldup_aggregate(circuit, f_con, f_lit, f_and, f_or, Union{Int32, Vector{Int32}})
    decisions2 = reshape(decisions, 2, :)
    elements2 = reshape(elements, 2, :) 
    parameters_nc = reshape(parameters, nc, :)
    @assert size(elements2)[2] == size(parameters_nc)[2]
    return ParameterCircuit(decisions2, elements2, parameters_nc)
end

struct CuParameterCircuit
    decisions::CuMatrix{Int32}
    elements::CuMatrix{Int32}
    parameters::CuMatrix{Float32}
    CuParameterCircuit(parameter_circuit::ParameterCircuit) = new(CuMatrix(parameter_circuit.decisions), CuMatrix(parameter_circuit.elements), CuMatrix(parameter_circuit.parameters))
end

function class_likelihood_and_flow(circuit::CuParameterCircuit, nc::Integer, data::CuMatrix{Float32}, reuse_up=nothing, reuse_down=nothing)
    cw, flow = class_weights_and_flow(circuit, nc, data, reuse_up, reuse_down)
    return @. 1.0 / (1.0 + exp(-cw)), flow
end

function class_weights_and_flow(circuit::CuParameterCircuit, nc::Integer, data::CuMatrix{Float32}, reuse_up=nothing, reuse_down=nothing)
    ne = num_examples(data)
    nf = num_features(data)
    nd = size(circuit.decisions)[2]
    nl = 2 + 2 * nf
    
    v = value_matrix(CuMatrix, ne, nl + nd, reuse_up)
    flow = if reuse_down isa CuArray{Float32} && size(reuse_down) == (ne, nl + nd)
        reuse_down
    else
        CUDA.zeros(Float32, ne, nl + nd)
    end
    cw = CUDA.zeros(Float32, ne, nc)
    
    set_leaf_layer(v, data)
    num_threads_per_block = 256
    numblocks = ceil(Int, ne/num_threads_per_block)
    CUDA.@sync begin
        @cuda threads=num_threads_per_block blocks=numblocks pass_down_and_sum_cw_layer_kernel_cuda(cw, flow, v, circuit.decisions, circuit.elements, circuit.parameters, ne, nc, nd, nl)
    end
    return cw, flow
end

function pass_down_and_sum_cw_layer_kernel_cuda(cw, flow, v, decisions, elements, parameters, ne, nc, nd, nl)
    evaluate_kernel_cuda(v, decisions, elements, ne, nd, nl)

    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for j = index:stride:ne
        flow[j, nl + nd] = v[j, nl + nd]
        for i = nd:-1:1
            first_elem = decisions[1, i]
            last_elem = decisions[2, i]
            n_up = v[j, nl + i]
            if n_up > zero(Float32)
                n_down = flow[j, nl + i]
                for k = first_elem:last_elem
                    e1 = elements[1, k]
                    e2 = elements[2, k]
                    c_up = v[j, e1] * v[j, e2]
                    additional_flow = c_up / n_up * n_down
                    flow[j, e1] += additional_flow
                    flow[j, e2] += additional_flow
                    for class = 1:nc
                        cw[j, class] += additional_flow * parameters[class, k]
                    end
                end
            end
        end
    end
    
    return nothing
end



struct ParameterLayer
    decisions::Matrix{Int32}
    elements::Matrix{Int32}
    parameters::Matrix{Float32}
end

struct LayeredParameterCircuit
    layers::Vector{ParameterLayer}
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
    
    f_con(n) = LayeredDecisionId(0, istrue(n) ? TRUE_BITS : FALSE_BITS)
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
                push!(elements[layer_id], c.decision_id, TRUE_BITS)
            end
        end
        push!(decisions[layer_id], num_decisions, first_element, num_elements[layer_id])
        LayeredDecisionId(layer_id, num_decisions)
    end

    foldup_aggregate(circuit, f_con, f_lit, f_and, f_or, 
        Union{LayeredDecisionId,Vector{LayeredDecisionId}})
    
    layers = map(decisions, elements, parameters) do d, e, p
        ParameterLayer(reshape(d, 3, :), reshape(e, 2, :), reshape(p, nc, :))
    end
    return LayeredParameterCircuit(layers)
end

struct CuParameterLayer
    decisions::CuMatrix{Int32}
    elements::CuMatrix{Int32}
    parameters::CuMatrix{Float32}
    CuParameterLayer(l::ParameterLayer) = new(CuMatrix(l.decisions), CuMatrix(l.elements), CuMatrix(l.parameters))
end

struct CuLayeredParameterCircuit
    layers::Vector{CuParameterLayer}
    CuLayeredParameterCircuit(l::LayeredParameterCircuit) = new(map(CuParameterLayer, l.layers))
end

num_decisions(l::CuParameterLayer) = size(l.decisions)[2]
num_decisions(l::CuLayeredParameterCircuit) = sum(num_decisions, l.layers)

function class_likelihood_and_flow2(circuit::CuLayeredParameterCircuit, nc::Integer, data::CuMatrix{Float32}, reuse_up=nothing, reuse_down=nothing)
    cw, flow = class_weights_and_flow2(circuit, nc, data, reuse_up, reuse_down)
    return @. 1.0 / (1.0 + exp(-cw)), flow
end

function class_weights_and_flow2(circuit::CuLayeredParameterCircuit, nc::Integer, data::CuMatrix{Float32}, reuse_up=nothing, reuse_down=nothing)
    ne = num_examples(data)
    nf = num_features(data)
    nd = num_decisions(circuit)
    nl = 2 + 2 * nf
    v = value_matrix(CuMatrix, ne, nl + nd, reuse_up)
    set_leaf_layer(v, data)
    
    dec_per_thread = 8
    CUDA.@sync for layer in circuit.layers
        ndl = num_decisions(layer)
        num_threads = balance_threads(ne, ndl / dec_per_thread, 8)
        num_blocks = (ceil(Int, ne / num_threads[1]), ceil(Int, ndl / num_threads[2] / dec_per_thread)) 
        @cuda threads=num_threads blocks=num_blocks evaluate_layer_kernel_cuda2(v, layer.decisions, layer.elements)
    end

    flow = if reuse_down isa CuArray{Float32} && size(reuse_down) == (ne, nl + nd)
        reuse_down .= zero(Float32)
        reuse_down
    else
        CUDA.zeros(Float32, ne, nl + nd)
    end
    flow[:,end] .= v[:,end] # set flow at root
    cw = CUDA.zeros(Float32, ne, nc) # initialize class_weights
    
    dec_per_thread = 4
    CUDA.@sync for layer in Iterators.reverse(circuit.layers)
        ndl = num_decisions(layer)
        num_threads = balance_threads(ne, ndl / dec_per_thread, 8)
        num_blocks = (ceil(Int, ne / num_threads[1]), ceil(Int, ndl / num_threads[2] / dec_per_thread)) 
        @cuda threads=num_threads blocks=num_blocks pass_down_and_sum_cw_layer_kernel_cuda2(cw, flow, v, layer.decisions, layer.elements, layer.parameters)
    end
    
    return cw, flow
end

function pass_down_and_sum_cw_layer_kernel_cuda2(cw, flow, v, decisions, elements, parameters)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().y
    ne, nc = size(cw)
    _, num_decisions = size(decisions)
    
    for j = index_x:stride_x:ne
        for i = index_y:stride_y:num_decisions
            decision_id = @inbounds decisions[1, i]
            first_elem = @inbounds decisions[2, i]
            last_elem = @inbounds decisions[3, i]
            n_up = @inbounds v[j, decision_id]
            if n_up > zero(Float32)
                n_down = @inbounds flow[j, decision_id]
                for e = first_elem:last_elem
                    e1 = @inbounds elements[1, e]
                    e2 = @inbounds elements[2, e]
                    c_up = @inbounds (v[j, e1] * v[j, e2])
                    additional_flow = c_up / n_up * n_down
                    # following needs to be memory safe
                    CUDA.@atomic flow[j, e1] += additional_flow #atomic is automatically inbounds
                    CUDA.@atomic flow[j, e2] += additional_flow #atomic is automatically inbounds
                    for class=1:nc
                        CUDA.@atomic cw[j, class] += additional_flow * parameters[class, e]
                    end
                end
            end
        end
    end
    
    return nothing
end



# TODO; paramter learning