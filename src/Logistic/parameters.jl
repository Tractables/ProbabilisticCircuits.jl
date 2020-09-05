export learn_parameters

using CUDA
using LoopVectorization: @avx, vifelse

"""
Parameter learning through gradient descents
"""
function learn_parameters(lc::LogisticCircuit, nc::Int, data, labels; num_epochs=25, step_size=0.01)
    bc = ParamBitCircuit(lc, nc, data)
    labels = one_hot(labels, nc)
    if isgpu(data)
        for _ = 1:num_epochs
            cl = class_likelihood_per_instance(bc, data)
            update_parameters_gpu(to_gpu(bc), cl, to_gpu(labels), step_size)
        end
    else
        for _ = 1:num_epochs
            cl = class_likelihood_per_instance(bc, data)
            update_parameters_cpu(bc, cw, labels, step_size)
        end
    end
end

function update_parameters_cpu(bc, cl, labels, step_size)
    ne::Int = num_examples(data)
    nc::Int = size(bc.params, 2)
    params_lock::Threads.ReentrantLock = Threads.ReentrantLock()

    @inline function on_edge_binary(flows, values, dec_id, el_id, p, s, els_start, els_end, locks)
        lock(cw_lock) do # TODO: move lock to inner loop?
            for i = 1:size(flows, 1)
                @inbounds edge_flow = values[i, p] & values[i, s] & flows[i, dec_id]
                first_true_bit = trailing_zeros(edge_flow) + 1
                last_true_bit = 64 - leading_zeros(edge_flow)
                @simd for j = first_true_bit:last_true_bit
                    if get_bit(edge_flow, j)
                        ex_id = ((i-1) << 6) + j
                        for class = 1:nc
                            @inbounds bc.params[el_id, class] -= (cl[ex_id, class] - labels[ex_id, class]) * step_size
                        end
                    end
                end
            end
        end
    end
    
    @inline function on_edge_float(flows, values, dec_id, el_id, p, s, els_start, els_end, locks)
        if els_start != els_end
            lock(cw_lock) do # TODO: move lock to inner loop?
                @avx for i = 1:size(flows, 1)
                    @inbounds edge_flow = values[i, p] * values[i, s] / values[i, dec_id] * flows[i, dec_id]
                    edge_flow = vifelse(isfinite(edge_flow), edge_flow, zero(Float32))
                    for class = 1:nc
                        @inbounds bc.parames[el_id, class] -= (cl[i, class] - labels[i, class]) * edge_flow * step_size
                    end
                end
            end
        end
        nothing
    end

    if isbinarydata(data)
        v,f = satisfies_flows(bc, data; on_edge = on_edge_binary) 
    else
        @assert isfpdata(data) "Only floating point and binary data are supported"
        v,f = satisfies_flows(bc, data; on_edge = on_edge_float)
    end

    nothing
end


function update_parameters_gpu(bc, cl, labels, step_size)
    ne::Int = num_examples(data)
    nc::Int = size(bc.params, 2)
    cl_device = CUDA.cudaconvert(cl)
    labels = CUDA.cudaconvert(labels)
    params_device = CUDA.cudaconvert(bc.params)

    @inline function on_edge_binary(flows, values, dec_id, el_id, p, s, els_start, els_end, chunk_id, edge_flow)
        if els_start != els_end
            first_true_bit = 1 + trailing_zeros(edge_flow)
            last_true_bit = 64 - leading_zeros(edge_flow)
            for j = first_true_bit:last_true_bit
                if get_bit(edge_flow, j)
                    ex_id = ((chunk_id-1) << 6) + j
                    for class = 1:nc
                        CUDA.@atomic params_device[el_id, class] -= (cl_device[ex_id, class] - labels[ex_id, class]) * step_size
                    end
                end
            end
        end
        nothing 
    end

    @inline function on_edge_float(flows, values, dec_id, el_id, p, s, els_start, els_end, ex_id, edge_flow)
        if els_start != els_end
            for class = 1:nc
                CUDA.@atomic params_device[el_id, class] -= (cl_device[ex_id, class] - lables[ex_id, class]) * edge_flow * step_size
            end
        end      
        nothing
    end

    if isbinarydata(data)
        v,f = satisfies_flows(bc, data; on_edge = on_edge_binary) 
    else
        @assert isfpdata(data) "Only floating point and binary data are supported"
        v,f = satisfies_flows(bc, data; on_edge = on_edge_float)
    end
    CUDA.unsafe_free!(v) # save the GC some effort
    CUDA.unsafe_free!(f) # save the GC some effort

    nothing
end



function one_hot(labels::Vector, nc::Integer)    
    ne = length(labels) 
    one_hot_labels = zeros(Float32, ne, nc)
    for (i, j) in enumerate(labels)
        one_hot_labels[i, j + 1] = 1.0
    end
    one_hot_labels
end