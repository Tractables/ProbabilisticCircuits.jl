export learn_parameters, to_onehot

using CUDA

"""
LogisticCircuit Parameter learning through gradient descents
Note: data need to be DataFrame and Labels need to be in one-hot form.
"""
function learn_parameters(lc::LogisticCircuit, nc::Int, data, labels; num_epochs=25, step_size=0.01)
    bc = ParamBitCircuit(lc, nc, data)
    if isgpu(data)
        @assert isgpu(labels) "Data and labels must be both stored in either GPU or CPU."
        for _ = 1:num_epochs
            cl = class_likelihood_per_instance(bc, data)
            update_parameters_gpu(to_gpu(bc), data, labels, cl, step_size)
        end
    else
        @assert !isgpu(labels) "Data and labels must be both stored in either GPU or CPU."
        for _ = 1:num_epochs
            cl = class_likelihood_per_instance(bc, data)
            update_parameters_cpu(bc, data, labels, cl, step_size)
        end
    end
end

function update_parameters_cpu(bc, data, labels, cl, step_size)
    ne::Int = num_examples(data)
    nc::Int = size(bc.params, 2)
    params_lock::Threads.ReentrantLock = Threads.ReentrantLock()

    @inline function on_edge_binary(flows, values, prime, sub, element, grandpa, single_child, weights)
        lock(params_lock) do # TODO: move lock to inner loop?
            for i = 1:size(flows, 1)
                @inbounds edge_flow = values[i, prime] & values[i, sub] & flows[i, grandpa]
                first_true_bit = trailing_zeros(edge_flow) + 1
                last_true_bit = 64 - leading_zeros(edge_flow)
                @simd for j = first_true_bit:last_true_bit
                    ex_id = ((i - 1) << 6) + j
                    if get_bit(edge_flow, j)
                        for class = 1:nc
                            @inbounds bc.params[element, class] -= (cl[ex_id, class] - labels[ex_id, class]) * step_size
                        end
                    end
                end
            end
        end
    end
    
    @inline function on_edge_float(flows, values, prime, sub, element, grandpa, single_child, weights)
        lock(params_lock) do # TODO: move lock to inner loop?
            @simd for i = 1:size(flows, 1) # adding @avx here might give incorrect results
                @inbounds edge_flow = values[i, prime] * values[i, sub] / values[i, grandpa] * flows[i, grandpa]
                edge_flow = ifelse(isfinite(edge_flow), edge_flow, zero(eltype(flows)))
                for class = 1:nc
                    @inbounds bc.parames[element, class] -= (cl[i, class] - labels[i, class]) * edge_flow * step_size
                end
            end
        end
        nothing
    end

    if isbinarydata(data)
        v,f = satisfies_flows(bc.bitcircuit, data; on_edge = on_edge_binary) 
    else
        @assert isfpdata(data) "Only floating point and binary data are supported"
        v,f = satisfies_flows(bc.bitcircuit, data; on_edge = on_edge_float)
    end

    nothing
end


function update_parameters_gpu(bc, data, labels, cl, step_size)
    ne::Int = num_examples(data)
    nc::Int = size(bc.params, 2)
    cl_device = CUDA.cudaconvert(cl)
    label_device = CUDA.cudaconvert(labels)
    params_device = CUDA.cudaconvert(bc.params)

    @inline function on_edge_binary(flows, values, prime, sub, element, grandpa, chunk_id, edge_flow, single_child, weights)
        first_true_bit = 1 + trailing_zeros(edge_flow)
        last_true_bit = 64 - leading_zeros(edge_flow)
        for j = first_true_bit:last_true_bit
            if get_bit(edge_flow, j)
                ex_id = ((chunk_id - 1) << 6) + j
                for class = 1:nc
                    CUDA.@atomic params_device[element, class] -= (cl_device[ex_id, class] - label_device[ex_id, class]) * step_size
                end
            end
        end
        nothing 
    end

    @inline function on_edge_float(flows, values, prime, sub, element, grandpa, ex_id, edge_flow, single_child, weights)
        for class = 1:nc
            CUDA.@atomic params_device[element, class] -= (cl_device[ex_id, class] - label_device[ex_id, class]) * edge_flow * step_size
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

    nothing
end



function to_onehot(labels::Vector, nc::Integer)    
    ne = length(labels) 
    one_hot_labels = zeros(Float32, ne, nc)
    for (i, j) in enumerate(labels)
        one_hot_labels[i, j + 1] = 1.0
    end
    one_hot_labels
end