export class_likelihood_per_instance, class_weights_per_instance

using CUDA
using LoopVectorization: @avx


"""
Class Conditional Probability
"""
function class_likelihood_per_instance(lc::LogisticCircuit, nc::Int, data)    
    cw = class_weights_per_instance(lc, nc, data)
    one = Float32(1.0)
    isgpu(data) ? (@. one / (one + exp(-cw))) : (@. @avx one / (one + exp(-cw)))
end

function class_likelihood_per_instance(bc, data)
    cw = class_weights_per_instance(bc, data)
    one = Float32(1.0)
    isgpu(data) ? (@. one / (one + exp(-cw))) : (@. @avx one / (one + exp(-cw)))
end

function class_weights_per_instance(lc::LogisticCircuit, nc::Int, data)
    bc = ParamBitCircuit(lc, nc, data)
    class_weights_per_instance(bc, data)
end

function class_weights_per_instance(bc, data)
    if isgpu(data)
        class_weights_per_instance_gpu(to_gpu(bc), data)
    else
        class_weights_per_instance_cpu(bc, data)
    end
end

function class_weights_per_instance_cpu(bc, data)
    ne::Int = num_examples(data)
    nc::Int = size(bc.params, 2)
    cw::Matrix{Float32} = zeros(Float32, ne, nc)
    cw_lock::Threads.ReentrantLock = Threads.ReentrantLock()
 
    @inline function on_edge_binary(flows, values, prime, sub, element, grandpa, single_child, weights)
        lock(cw_lock) do # TODO: move lock to inner loop?
            for i = 1:size(flows, 1)
                @inbounds edge_flow = values[i, prime] & values[i, sub] & flows[i, grandpa]
                first_true_bit = trailing_zeros(edge_flow) + 1
                last_true_bit = 64 - leading_zeros(edge_flow)
                @simd for j = first_true_bit:last_true_bit
                    ex_id = ((i - 1) << 6) + j
                    if get_bit(edge_flow, j)
                        for class = 1:nc
                            @inbounds cw[ex_id, class] += bc.params[element, class]
                        end
                    end
                end
            end
        end
        nothing
    end

    @inline function on_edge_float(flows, values, prime, sub, element, grandpa, single_child, weights)
        lock(cw_lock) do # TODO: move lock to inner loop?
            @simd for i = 1:size(flows, 1) # adding @avx here might give incorrect results
                @inbounds edge_flow = values[i, prime] * values[i, sub] / values[i, grandpa] * flows[i, grandpa]
                edge_flow = ifelse(isfinite(edge_flow), edge_flow, zero(eltype(flows)))
                for class = 1:nc
                    @inbounds cw[i, class] += edge_flow * bc.params[element, class]
                end
            end
        end
        nothing
    end

    if isbinarydata(data)
        satisfies_flows(bc.bitcircuit, data; on_edge = on_edge_binary)
    else
        satisfies_flows(bc.bitcircuit, data; on_edge = on_edge_float)
    end

    return cw
end

function class_weights_per_instance_gpu(bc, data)
    ne::Int = num_examples(data)
    nc::Int = size(bc.params, 2)
    cw::CuMatrix{Float32} = CUDA.zeros(Float32, num_examples(data), nc)
    cw_device = CUDA.cudaconvert(cw)
    params_device = CUDA.cudaconvert(bc.params)

    @inline function on_edge_binary(flows, values, prime, sub, element, grandpa, chunk_id, edge_flow, single_child, weights)
        first_true_bit = 1 + trailing_zeros(edge_flow)
        last_true_bit = 64 - leading_zeros(edge_flow)
        for j = first_true_bit:last_true_bit
            ex_id = ((chunk_id - 1) << 6) + j
            if get_bit(edge_flow, j)
                for class = 1:nc
                    CUDA.@atomic cw_device[ex_id, class] += params_device[element, class]
                end
            end
        end
        nothing
    end

    @inline function on_edge_float(flows, values, prime, sub, element, grandpa, ex_id, edge_flow, single_child, weights)
        for class = 1:nc
            CUDA.@atomic cw_device[ex_id, class] += edge_flow * params_device[element, class]
        end
        nothing
    end
    
    if isbinarydata(data)
        v,f = satisfies_flows(bc.bitcircuit, data; on_edge = on_edge_binary)
    else
        @assert isfpdata(data) "Only floating point and binary data are supported"
        v,f = satisfies_flows(bc.bitcircuit, data; on_edge = on_edge_float)
    end
    CUDA.unsafe_free!(v) # save the GC some effort
    CUDA.unsafe_free!(f) # save the GC some effort

    return cw
end



"""
Class Predictions
"""
function predict_class(lc::LogisticCircuit, nc::Int, data)
    class_likelihoods = class_likelihood_per_instance(lc, nc, data)
    predict_class(class_likelihoods)
end

function predict_class(class_likelihoods)
    _, mxindex = findmax(class_likelihoods; dims=2)
    dropdims(getindex.(mxindex, 2); dims=2)
end



"""
Prediction accuracy
"""
accuracy(lc::LogisticCircuit, nc::Int, data, labels::Vector) = 
    accuracy(predict_class(lc, nc, data), labels)

accuracy(predicted_class::Vector, labels::Vector) = 
    Float64(sum(@. predicted_class == labels)) / length(labels)

accuracy(class_likelihoods::Matrix, labels::Vector) = 
    accuracy(predict_class(class_likelihoods), labels)
