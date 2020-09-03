export EVI, log_likelihood_per_instance, log_likelihood, log_likelihood_avg

"""
Construct a `BitCircuit` while storing edge parameters in a separate array
"""
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

"""
Compute the likelihood of the PC given each individual instance in the data
"""
function log_likelihood_per_instance(pc::ProbCircuit, data)
    @assert isbinarydata(data) "Probabilistic circuit likelihoods are for binary data only"
    bc = ParamBitCircuit(pc, data)
    if isgpu(data)
        log_likelihood_per_instance_gpu(to_gpu(bc), data)
    else
        log_likelihood_per_instance_cpu(bc, data)
    end
end

function log_likelihood_per_instance_cpu(bc, data)
    ll::Vector{Float64} = zeros(Float64, num_examples(data))
    ll_lock::Threads.ReentrantLock = Threads.ReentrantLock()
    
    @inline function on_edge(flows, values, dec_id, el_id, p, s, els_start, els_end, locks)
        if els_start != els_end
            lock(ll_lock) do # TODO: move lock to inner loop?
                for i = 1:size(flows,1)
                    @inbounds edge_flow = values[i, p] & values[i, s] & flows[i, dec_id]
                    first_true_bit = trailing_zeros(edge_flow)+1
                    last_true_bit = 64-leading_zeros(edge_flow)
                    @simd for j = first_true_bit:last_true_bit
                        ex_id = ((i-1) << 6) + j
                        if get_bit(edge_flow, j)
                            @inbounds ll[ex_id] += bc.params[el_id]
                        end
                    end
                end
            end
        end
        nothing
    end

    compute_values_flows(bc.bitcircuit, data; on_edge)
    return ll
end

function log_likelihood_per_instance_gpu(bc, data)
    params_device = CUDA.cudaconvert(bc.params)
    ll::CuVector{Float64} = CUDA.zeros(Float64, num_examples(data))
    ll_device = CUDA.cudaconvert(ll)
        
    @inline function on_edge(flows, values, dec_id, el_id, p, s, els_start, els_end, chunk_id, edge_flow)
        if els_start != els_end
            first_true_bit = 1+trailing_zeros(edge_flow)
            last_true_bit = 64-leading_zeros(edge_flow)
            for j = first_true_bit:last_true_bit
                ex_id = ((chunk_id-1) << 6) + j
                if get_bit(edge_flow, j)
                    CUDA.@atomic ll_device[ex_id] += params_device[el_id]
                end
            end
        end
        nothing
    end
    
    v, f = compute_values_flows(bc.bitcircuit, data; on_edge)
    CUDA.unsafe_free!(v) # save the GC some effort
    CUDA.unsafe_free!(f) # save the GC some effort
    
    return ll
end

"""
Complete evidence queries
"""
EVI = log_likelihood_per_instance

"""
Compute the likelihood of the PC given the data
"""
log_likelihood(pc, data) = sum(log_likelihood_per_instance(pc, data))

"""
Compute the likelihood of the PC given the data, averaged over all instances in the data
"""
log_likelihood_avg(pc, data) = log_likelihood(pc, data)/num_examples(data)