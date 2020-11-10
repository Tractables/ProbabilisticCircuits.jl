using DataFrames

export EVI, log_likelihood_per_instance, log_likelihood, log_likelihood_avg

"""
Compute the likelihood of the PC given each individual instance in the data
"""
function log_likelihood_per_instance(pc::ProbCircuit, data; use_gpu::Bool = false)
    log_likelihood_per_instance_single_model(pc, data; use_gpu)
end
function log_likelihood_per_instance(spc::SharedProbCircuit, data; use_gpu::Bool = false)
    get_single_model_ll(component_idx::Integer) = begin
        log_likelihood_per_instance_single_model(spc, data; use_gpu, component_idx)
    end
    
    mapreduce(get_single_model_ll, +, [1:num_components(spc);]) ./ num_components(spc)
end
function log_likelihood_per_instance_single_model(pc::ProbCircuit, data; use_gpu::Bool = false, component_idx::Integer = 0)
    @assert isbinarydata(data) "Probabilistic circuit likelihoods are for binary data only"
    if pc isa SharedProbCircuit
        bc = ParamBitCircuit(pc, data; component_idx)
    else
        bc = ParamBitCircuit(pc, data)
    end
    if isgpu(data)
        use_gpu = true
    end
    if use_gpu
        log_likelihood_per_instance_gpu(to_gpu(bc), to_gpu(data))
    else
        log_likelihood_per_instance_cpu(bc, data)
    end
end

function log_likelihood_per_instance_cpu(bc, data)
    ll::Vector{Float64} = zeros(Float64, num_examples(data))
    ll_lock::Threads.ReentrantLock = Threads.ReentrantLock()
    
    @inline function on_edge(flows, values, prime, sub, element, grandpa, single_child, weights)
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
        
    @inline function on_edge(flows, values, prime, sub, element, grandpa, chunk_id, edge_flow, single_child, weights)
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


function log_likelihood_per_instance_reuse(bc::ParamBitCircuit, data, reuse_v, reuse_f; use_gpu::Bool = false)
    @assert isbinarydata(data) "Probabilistic circuit likelihoods are for binary data only"
    if isgpu(data)
        use_gpu = true
    end
    if use_gpu
        log_likelihood_per_instance_reuse_gpu(to_gpu(bc), to_gpu(data), reuse_v, reuse_f)
    else
        log_likelihood_per_instance_reuse_cpu(bc, data, reuse_v, reuse_f)
    end
end

function log_likelihood_per_instance_reuse_cpu(bc, data, reuse_v, reuse_f)
    ll::Vector{Float64} = zeros(Float64, num_examples(data))
    ll_lock::Threads.ReentrantLock = Threads.ReentrantLock()
    
    @inline function on_edge(flows, values, prime, sub, element, grandpa, single_child, weights)
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

    v, f = satisfies_flows(bc.bitcircuit, data, reuse_v, reuse_f; on_edge)
    
    # when the example is outside of the support, give 0 likelihood 
    in_support = AbstractBitVector(v[:,end], num_examples(data))
    ll[.! in_support] .= -Inf

    return ll, v, f
end

function log_likelihood_per_instance_reuse_gpu(bc, data, reuse_v, reuse_f)
    params_device = CUDA.cudaconvert(bc.params)
    ll::CuVector{Float64} = CUDA.zeros(Float64, num_examples(data))
    ll_device = CUDA.cudaconvert(ll)
        
    @inline function on_edge(flows, values, prime, sub, element, grandpa, chunk_id, edge_flow, single_child, weights)
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
    
    v, f = satisfies_flows(bc.bitcircuit, data, reuse_v, reuse_f; on_edge)

    # when the example is outside of the support, give 0 likelihood 
    # lazy programmer: do the conversion to a Vector{Bool} on CPU 
    #   so that CUDA.jl can build a quick kernel
    # TODO: write a function to do this conversion on GPU directly
    in_support = AbstractBitVector(to_cpu(v[:,end]), num_examples(data))
    in_support = to_gpu(convert(Vector{Bool}, in_support))
    ll2 = map((x,s) -> s ? x : -Inf, ll, in_support)
    
    CUDA.unsafe_free!(ll) # save the GC some effort
    
    return ll2, v, f
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
log_likelihood(pc, data; use_gpu::Bool = false) = begin
    if isweighted(data)
        # `data' is weighted according to its `weight' column
        data, weights = split_sample_weights(data)
        
        log_likelihood(pc, data, weights; use_gpu)
    else
        likelihoods = log_likelihood_per_instance(pc, data; use_gpu)
        
        isgpu(likelihoods) ? sum(to_cpu(likelihoods)) : sum(likelihoods)
    end
end
log_likelihood(pc, data, weights::DataFrame; use_gpu::Bool = false) = log_likelihood(pc, data, weights[:, 1]; use_gpu)
log_likelihood(pc, data, weights::AbstractArray; use_gpu::Bool = false) = begin
    if isgpu(weights)
        weights = to_cpu(weights)
    end
    likelihoods = log_likelihood_per_instance(pc, data; use_gpu)
    if isgpu(likelihoods)
        likelihoods = to_cpu(likelihoods)
    end
    mapreduce(*, +, likelihoods, weights)
end
log_likelihood(pc, data::Array{DataFrame}; use_gpu::Bool = false) = begin
    if pc isa SharedProbCircuit
        total_ll = 0.0
        for component_idx = 1 : num_components(pc)
            total_ll += log_likelihood_batched(pc, data; use_gpu, component_idx)
        end
        total_ll / num_components(pc)
    else
        log_likelihood_batched(pc, data; use_gpu)
    end
end
log_likelihood_batched(pc, data::Array{DataFrame}; use_gpu::Bool = false, component_idx::Integer = 0) = begin
    # mapreduce(d -> log_likelihood(pc, d; use_gpu), +, data)
    if pc isa SharedProbCircuit
        pbc = ParamBitCircuit(pc, data; component_idx)
    else
        pbc = ParamBitCircuit(pc, data)
    end
    
    total_ll::Float64 = 0.0
    if isweighted(data)
        data, weights = split_sample_weights(data)
        
        v, f = nothing, nothing
        for idx = 1 : length(data)
            likelihoods, v, f = log_likelihood_per_instance_reuse(pbc, data[idx], v, f; use_gpu)
            if isgpu(likelihoods)
                likelihoods = to_cpu(likelihoods)
            end
            w = weights[idx]
            if isgpu(w)
                w = to_cpu(w)
            end
            total_ll += mapreduce(*, +, likelihoods, w)
        end
    else
        v, f = nothing, nothing
        for idx = 1 : length(data)
            likelihoods, v, f = log_likelihood_per_instance_reuse(pbc, data[idx], v, f; use_gpu)
            total_ll += sum(isgpu(likelihoods) ? to_cpu(likelihoods) : likelihoods)
        end
    end
    
    if use_gpu
        CUDA.unsafe_free!(v) # save the GC some effort
        CUDA.unsafe_free!(f) # save the GC some effort
    end
    
    total_ll
end

"""
    log_likelihood_avg(pc, data)

Compute the likelihood of the PC given the data, averaged over all instances in the data
"""
log_likelihood_avg(pc, data; use_gpu::Bool = false) = begin
    if isweighted(data)
        # `data' is weighted according to its `weight' column
        data, weights = split_sample_weights(data)
        
        log_likelihood_avg(pc, data, weights; use_gpu)
    else
        log_likelihood(pc, data; use_gpu) / num_examples(data)
    end
end
log_likelihood_avg(pc, data, weights::DataFrame; use_gpu::Bool = false) = log_likelihood_avg(pc, data, weights[:, 1]; use_gpu)
log_likelihood_avg(pc, data, weights; use_gpu::Bool = false) = begin
    if isgpu(weights)
        weights = to_cpu(weights)
    end
    log_likelihood(pc, data, weights; use_gpu) / sum(weights)
end
log_likelihood_avg(pc, data::Array{DataFrame}; use_gpu::Bool = false) = begin
    if isweighted(data)
        weights = get_weights(data)
        if isgpu(weights)
            weights = to_cpu(weights)
        end
        log_likelihood(pc, data; use_gpu) / mapreduce(sum, +, weights)
    else
        log_likelihood(pc, data; use_gpu) / num_examples(data)
    end
end