using CUDA
using LogicCircuits

export ParameterCircuit, CuParameterCircuit
export class_likelihood_and_flow, class_weights_and_flow

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

function class_likelihood_and_flow(parameter_circuit::CuParameterCircuit, nc::Integer, data::CuMatrix{Float32}, reuse_up=nothing, reuse_down=nothing)
    cw, flow = class_weights_and_flow(parameter_circuit, nc, data, reuse_up, reuse_down)
    return @. 1.0 / (1.0 + exp(-cw)), flow
end

function class_weights_and_flow(parameter_circuit::CuParameterCircuit, nc::Integer, data::CuMatrix{Float32}, reuse_up=nothing, reuse_down=nothing)
    ne = num_examples(data)
    nf = num_features(data)
    nd = size(parameter_circuit.decisions)[2]
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
        @cuda threads=num_threads_per_block blocks=numblocks class_weights_flow_kernel_cuda(cw, flow, v, parameter_circuit.decisions, parameter_circuit.elements, parameter_circuit.parameters, ne, nc, nd, nl)
    end
    return cw, flow
end

function class_weights_flow_kernel_cuda(cw, flow, v, decisions, elements, parameters, ne, nc, nd, nl)
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