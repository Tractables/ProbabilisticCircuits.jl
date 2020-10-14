using DataFrames

export EVI, log_likelihood_per_instance, log_likelihood, log_likelihood_avg

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
    
    @inline function on_edge(flows, values, prime, sub, element, grandpa, single_child)
        if !single_child
            lock(ll_lock) do # TODO: move lock to inner loop? change to atomic float?
                for i = 1:size(flows,1)
                    @inbounds edge_flow = values[i, prime] & values[i, sub] & flows[i, grandpa]
                    first_true_bit = trailing_zeros(edge_flow)+1
                    last_true_bit = 64-leading_zeros(edge_flow)
                    @simd for j = first_true_bit:last_true_bit
                        ex_id = ((i-1) << 6) + j
                        if get_bit(edge_flow, j)
                            @inbounds ll[ex_id] += bc.params[element]
                        end
                    end
                end
            end
        end
        nothing
    end

    v, _ = satisfies_flows(bc.bitcircuit, data; on_edge)
    
    # when the example is outside of the support, give 0 likelihood 
    in_support = AbstractBitVector(v[:,end], num_examples(data))
    ll[.! in_support] .= -Inf

    return ll
end

function log_likelihood_per_instance_gpu(bc, data)
    params_device = CUDA.cudaconvert(bc.params)
    ll::CuVector{Float64} = CUDA.zeros(Float64, num_examples(data))
    ll_device = CUDA.cudaconvert(ll)
        
    @inline function on_edge(flows, values, prime, sub, element, grandpa, chunk_id, edge_flow, single_child)
        if !single_child
            first_true_bit = 1+trailing_zeros(edge_flow)
            last_true_bit = 64-leading_zeros(edge_flow)
            for j = first_true_bit:last_true_bit
                ex_id = ((chunk_id-1) << 6) + j
                if get_bit(edge_flow, j)
                    CUDA.@atomic ll_device[ex_id] += params_device[element]
                end
            end
        end
        nothing
    end
    
    v, f = satisfies_flows(bc.bitcircuit, data; on_edge)
    CUDA.unsafe_free!(f) # save the GC some effort

    # when the example is outside of the support, give 0 likelihood 
    # lazy programmer: do the conversion to a Vector{Bool} on CPU 
    #   so that CUDA.jl can build a quick kernel
    # TODO: write a function to do this conversion on GPU directly
    in_support = AbstractBitVector(to_cpu(v[:,end]), num_examples(data))
    in_support = to_gpu(convert(Vector{Bool}, in_support))
    ll2 = map((x,s) -> s ? x : -Inf, ll, in_support)
    CUDA.unsafe_free!(v) # save the GC some effort
    CUDA.unsafe_free!(ll) # save the GC some effort
    
    return ll2
end

"""
    EVI(pc, data)

Computes the log likelihood data given full evidence.
Outputs `` \\log{p(x)} `` for each datapoint.
"""
const EVI = log_likelihood_per_instance

"""
    log_likelihood(pc, data)

Compute the likelihood of the PC given the data
"""
log_likelihood(pc, data) = begin
    if isweighted(data)
        # `data' is weighted according to its `weight' column
        weights = data[:, end]
        data = data[:, 1:end - 1]
        
        log_likelihood(pc, data, weights)
    else
        sum(log_likelihood_per_instance(pc, data))
    end
end
log_likelihood(pc, data, weights::DataFrame) = log_likelihood(pc, data, weights[:, 1])
log_likelihood(pc, data, weights::AbstractArray) = begin
    if isgpu(weights)
        weights = to_cpu(weights)
    end
    likelihoods = log_likelihood_per_instance(pc, data)
    if isgpu(likelihoods)
        likelihoods = to_cpu(likelihoods)
    end
    mapreduce(*, +, likelihoods, weights)
end

"""
    log_likelihood_avg(pc, data)

Compute the likelihood of the PC given the data, averaged over all instances in the data
"""
log_likelihood_avg(pc, data) = begin
    if isweighted(data)
        # `data' is weighted according to its `weight' column
        weights = data[:, end]
        data = data[:, 1:end - 1]
        
        log_likelihood_avg(pc, data, weights)
    else
        log_likelihood(pc, data)/num_examples(data)
    end
end
log_likelihood_avg(pc, data, weights::DataFrame) = log_likelihood_avg(pc, data, weights[:, 1])
log_likelihood_avg(pc, data, weights) = begin
    if isgpu(weights)
        weights = to_cpu(weights)
    end
    log_likelihood(pc, data, weights)/sum(weights)
end