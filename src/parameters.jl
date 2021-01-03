export estimate_parameters, uniform_parameters, estimate_parameters_em, estimate_parameters_cached!, 
       apply_entropy_reg

using StatsFuns: logsumexp, logaddexp
using CUDA
using LoopVectorization

"""
Maximum likilihood estimation of parameters given data

use_gpu: If set to `true`, use gpu learning no matter which device `data` is in.

bagging support: If `pc` is a SharedProbCircuit and data is an array of DataFrames
  with the same number of "components", learn each circuit with its corresponding 
  dataset.
"""
function estimate_parameters(pc::ProbCircuit, data; pseudocount::Float64,
                             use_sample_weights::Bool = true, use_gpu::Bool = false, entropy_reg::Float64 = 0.0)
    params = estimate_single_circuit_parameters(pc, data; pseudocount, use_sample_weights, use_gpu)
    
    if entropy_reg > 1e-9
        apply_entropy_reg(pc; alpha = entropy_reg)
    end
    
    params
end
function estimate_parameters(spc::SharedProbCircuit, data; pseudocount::Float64,
                             use_sample_weights::Bool = true, use_gpu::Bool = false)
    @assert num_components(spc) == length(data) "SharedProbCircuit and data have different number of components: $(num_components(spc)) and $(length(data)), resp."
    
    map(1 : num_components(spc)) do component_idx
        estimate_single_circuit_parameters(spc, data[component_idx]; pseudocount, use_sample_weights, use_gpu, component_idx)
    end
end
function estimate_single_circuit_parameters(pc::ProbCircuit, data; pseudocount::Float64, 
                                            use_sample_weights::Bool = true, use_gpu::Bool = false,
                                            component_idx::Integer = 0)
    if isweighted(data)
        # `data' is weighted according to its `weight' column
        data, weights = split_sample_weights(data)
    else
        use_sample_weights = false
    end
    
    @assert isbinarydata(data) || isfpdata(data) "Probabilistic circuit parameter estimation for binary/floating point data only"
    bc = BitCircuit(pc, data; reset=false)
    if isgpu(data)
        use_gpu = true
    end
    params = if use_gpu
        if use_sample_weights
            estimate_parameters_gpu(to_gpu(bc), data, pseudocount; weights)
        else
            estimate_parameters_gpu(to_gpu(bc), data, pseudocount)
        end
    else
        if use_sample_weights
            estimate_parameters_cpu(bc, data, pseudocount; weights)
        else
            estimate_parameters_cpu(bc, data, pseudocount)
        end
    end
    
    if pc isa SharedProbCircuit
        estimate_parameters_cached!(pc, bc, params, component_idx)
    else
        estimate_parameters_cached!(pc, bc, params)
    end
    params
end

function estimate_parameters_cached!(pc::SharedProbCircuit, bc, params, component_idx; exp_update_factor = 0.0)
    log_exp_factor = log(exp_update_factor)
    log_1_exp_factor = log(1.0 - exp_update_factor)
    foreach_reset(pc) do pn
        if is⋁gate(pn)
            if num_children(pn) == 1
                pn.log_probs[:, component_idx] .= zero(Float64)
            else
                id = (pn.data::⋁NodeIds).node_id
                @inbounds els_start = bc.nodes[1,id]
                @inbounds els_end = bc.nodes[2,id]
                @inbounds @views pn.log_probs[:, component_idx] .= logaddexp.(log_exp_factor .+ pn.log_probs[:, component_idx], log_1_exp_factor .+ params[els_start:els_end])
                @assert isapprox(sum(exp.(pn.log_probs[:, component_idx])), 1.0, atol=1e-3) "Parameters do not sum to one locally: $(sum(exp.(pn.log_probs))); $(pn.log_probs)"
                pn.log_probs[:, component_idx] .-= logsumexp(pn.log_probs[:, component_idx]) # normalize away any leftover error
            end
        end
    end
    nothing
end
function estimate_parameters_cached!(pc::ProbCircuit, pbc; exp_update_factor = 0.0)
    if isgpu(pbc)
        pbc = to_cpu(pbc)
    end
    bc = pbc.bitcircuit
    params = pbc.params
    estimate_parameters_cached!(pc, bc, params; exp_update_factor)
end
function estimate_parameters_cached!(pc::ProbCircuit, bc, params; exp_update_factor = 0.0)
    log_exp_factor = log(exp_update_factor)
    log_1_exp_factor = log(1.0 - exp_update_factor)
    
    foreach_reset(pc) do pn
        if is⋁gate(pn)
            if num_children(pn) == 1
                pn.log_probs .= zero(Float64)
            else
                id = (pn.data::⋁NodeIds).node_id
                @inbounds els_start = bc.nodes[1,id]
                @inbounds els_end = bc.nodes[2,id]
                @inbounds @views pn.log_probs .= logaddexp.(log_exp_factor .+ pn.log_probs, log_1_exp_factor .+ params[els_start:els_end])
                @assert isapprox(sum(exp.(pn.log_probs)), 1.0, atol=1e-3) "Parameters do not sum to one locally: $(sum(exp.(pn.log_probs))); $(pn.log_probs)"
                pn.log_probs .-= logsumexp(pn.log_probs) # normalize away any leftover error
            end
        end
    end
    
    nothing
end

function estimate_parameters_cpu(bc::BitCircuit, data, pseudocount; weights = nothing)
    # no need to synchronize, since each computation is unique to a decision node
    if weights === nothing && isbinarydata(data)
        node_counts = Vector{UInt}(undef, num_nodes(bc))
    else
        node_counts = Vector{Float64}(undef, num_nodes(bc))
    end
    
    if isbatched(data)
        estimate_parameters_cpu_batched(bc, data, pseudocount, node_counts; weights = weights)
    else
        estimate_parameters_cpu_not_batched(bc, data, pseudocount, node_counts; weights = weights)
    end
end
function estimate_parameters_cpu_batched(bc::BitCircuit, data, pseudocount, node_counts; weights = nothing)
    # rescale pseudocount using the average weight of samples
    if weights !== nothing
        pseudocount = pseudocount * mapreduce(sum, +, weights) / num_examples(data)
    end
    
    edge_counts::Vector{Float64} = zeros(Float64, num_elements(bc))
    parent_node_counts::Vector{Float64} = zeros(Float64, num_elements(bc))
    
    @inline function weighted_count_ones(bits::UInt64, start_idx::Number, end_idx::Number, weights)
        count::Float64 = 0.0
        op_bits::UInt64 = bits
        @inbounds for idx = 0 : (end_idx - start_idx)
            count += weights[start_idx + idx] * (op_bits & UInt64(0x1))
            op_bits = (op_bits >> 1)
        end
        count
    end

    @inline function on_node(flows, values, dec_id, weights::Nothing)
        node_counts[dec_id] = sum(1:size(flows,1)) do i
            count_ones(flows[i, dec_id]) 
        end
    end
    @inline function on_node(flows, values, dec_id, weights)
        node_counts[dec_id] = sum(1:size(flows,1)) do i
            weighted_count_ones(flows[i, dec_id], i * 64 - 63, min(i * 64, length(weights)), weights)
        end
    end
    @inline function on_node(flows, values::Matrix{<:AbstractFloat}, dec_id, weights::Nothing)
        node_counts[dec_id] = sum(1:size(flows,1)) do i
            flows[i, dec_id]
        end
    end
    @inline function on_node(flows, values::Matrix{<:AbstractFloat}, dec_id, weights)
        node_counts[dec_id] = sum(1:size(flows,1)) do i
            flows[i, dec_id] * weights[i]
        end
    end

    @inline function estimate(element, decision, edge_count)
        edge_counts[element] += edge_count
        parent_node_counts[element] += node_counts[decision]
    end

    @inline function on_edge(flows, values, prime, sub, element, grandpa, single_child, weights::Nothing)
        if !single_child
            edge_count = sum(1:size(flows,1)) do i
                count_ones(values[i, prime] & values[i, sub] & flows[i, grandpa]) 
            end
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end
    @inline function on_edge(flows, values, prime, sub, element, grandpa, single_child, weights)
        if !single_child
            edge_count = sum(1:size(flows,1)) do i
                weighted_count_ones(values[i, prime] & values[i, sub] & flows[i, grandpa], 
                                    i * 64 - 63, min(i * 64, length(weights)), weights)
            end
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end
    @inline function on_edge(flows, values::Matrix{<:AbstractFloat}, prime, sub, element, grandpa, single_child, weights::Nothing)
        if !single_child
            edge_count = sum(1:size(flows,1)) do i
                values[i, prime] * values[i, sub] * flows[i, grandpa]
            end
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end
    @inline function on_edge(flows, values::Matrix{<:AbstractFloat}, prime, sub, element, grandpa, single_child, weights)
        if !single_child
            edge_count = sum(1:size(flows,1)) do i
                values[i, prime] * values[i, sub] * flows[i, grandpa] * weights[i]
            end
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end

    v, f = nothing, nothing
    if weights != nothing
        map(zip(data, weights)) do (d, w)
            v, f = satisfies_flows(bc, d, v, f; on_node = on_node, on_edge = on_edge, weights = w)
        end
    else
        map(data) do d
            v, f = satisfies_flows(bc, d, v, f; on_node = on_node, on_edge = on_edge, weights = nothing)
        end
    end
    
    # Reuse `edge_counts` to store log_params to save space and time.
    for i = 1 : num_elements(bc)
        @inbounds edge_counts[i] = log((edge_counts[i] + pseudocount / num_elements(bc.nodes, bc.elements[1, i])) / (parent_node_counts[i] + pseudocount))
    end

    edge_counts # a.k.a. log_params
end
function estimate_parameters_cpu_not_batched(bc::BitCircuit, data, pseudocount, node_counts; weights = nothing)
    # rescale pseudocount using the average weight of samples
    if weights !== nothing
        pseudocount = pseudocount * sum(weights) / size(weights, 1)
    end
    
    log_params::Vector{Float64} = Vector{Float64}(undef, num_elements(bc))
    
    @inline function weighted_count_ones(bits::UInt64, start_idx::Number, end_idx::Number)
        count::Float64 = 0.0
        op_bits::UInt64 = bits
        @inbounds for idx = 0 : (end_idx - start_idx)
            count += weights[start_idx + idx] * (op_bits & UInt64(0x1))
            op_bits = (op_bits >> 1)
        end
        count
    end

    @inline function on_node(flows, values, dec_id, weights::Nothing)
        node_counts[dec_id] = sum(1:size(flows,1)) do i
            count_ones(flows[i, dec_id]) 
        end
    end
    @inline function on_node(flows, values, dec_id, weights)
        node_counts[dec_id] = sum(1:size(flows,1)) do i
            weighted_count_ones(flows[i, dec_id], i * 64 - 63, min(i * 64, length(weights)))
        end
    end
    @inline function on_node(flows, values::Matrix{<:AbstractFloat}, dec_id, weights::Nothing)
        node_counts[dec_id] = sum(1:size(flows,1)) do i
            flows[i, dec_id]
        end
    end
    @inline function on_node(flows, values::Matrix{<:AbstractFloat}, dec_id, weights)
        node_counts[dec_id] = sum(1:size(flows,1)) do i
            flows[i, dec_id] * weights[i]
        end
    end

    @inline function estimate(element, decision, edge_count)
        num_els = num_elements(bc.nodes, decision)
        log_params[element] = 
            log((edge_count+pseudocount/num_els)
                   /(node_counts[decision]+pseudocount))
    end

    @inline function on_edge(flows, values, prime, sub, element, grandpa, single_child, weights::Nothing)
        if !single_child
            edge_count = sum(1:size(flows,1)) do i
                count_ones(values[i, prime] & values[i, sub] & flows[i, grandpa]) 
            end
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end
    @inline function on_edge(flows, values, prime, sub, element, grandpa, single_child, weights)
        if !single_child
            edge_count = sum(1:size(flows,1)) do i
                weighted_count_ones(values[i, prime] & values[i, sub] & flows[i, grandpa], 
                                    i * 64 - 63, min(i * 64, length(weights)))
            end
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end
    @inline function on_edge(flows, values::Matrix{<:AbstractFloat}, prime, sub, element, grandpa, single_child, weights::Nothing)
        if !single_child
            edge_count = sum(1:size(flows,1)) do i
                values[i, prime] * values[i, sub] * flows[i, grandpa]
            end
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end
    @inline function on_edge(flows, values::Matrix{<:AbstractFloat}, prime, sub, element, grandpa, single_child, weights)
        if !single_child
            edge_count = sum(1:size(flows,1)) do i
                values[i, prime] * values[i, sub] * flows[i, grandpa] * weights[i]
            end
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end

    v, f = satisfies_flows(bc, data; on_node, on_edge, weights)

    log_params
end

function estimate_parameters_gpu(bc::BitCircuit, data, pseudocount; weights = nothing)
    # rescale pseudocount using the average weight of samples
    if weights !== nothing
        if isbatched(data)
            pseudocount = pseudocount * mapreduce(sum, +, weights) / num_examples(data)
        else
            pseudocount = pseudocount * sum(weights) / size(weights, 1)
        end
    end
    
    node_counts::CuVector{Float32} = CUDA.zeros(Float32, num_nodes(bc))
    edge_counts::CuVector{Float32} = CUDA.zeros(Float32, num_elements(bc))
    # need to manually cudaconvert closure variables
    node_counts_device = CUDA.cudaconvert(node_counts)
    edge_counts_device = CUDA.cudaconvert(edge_counts)
    
    @inline function on_node(flows, values, dec_id, chunk_id, flow, weight::Nothing)
        c::Float32 = CUDA.count_ones(flow) # cast for @atomic to be happy
        CUDA.@atomic node_counts_device[dec_id] += c
    end
    @inline function on_node(flows, values, dec_id, bit_idx, flow, weight::Float32)
        c::Float32 = ((flow >> bit_idx) & UInt64(0x1)) * weight # cast for @atomic to be happy
        CUDA.@atomic node_counts_device[dec_id] += c
    end
    @inline function on_node(flows, values, dec_id, bit_idx, flow::AbstractFloat, weight::Nothing)
        c::Float32 = flow # cast for @atomic to be happy
        CUDA.@atomic node_counts_device[dec_id] += c
    end
    @inline function on_node(flows, values, dec_id, bit_idx, flow::AbstractFloat, weight::Float32)
        c::Float32 = flow * weight # cast for @atomic to be happy
        CUDA.@atomic node_counts_device[dec_id] += c
    end

    @inline function on_edge(flows, values, prime, sub, element, grandpa, chunk_id, edge_flow, single_child, weight::Nothing)
        if !single_child
            c::Float32 = CUDA.count_ones(edge_flow) # cast for @atomic to be happy
            CUDA.@atomic edge_counts_device[element] += c
        end
    end
    @inline function on_edge(flows, values, prime, sub, element, grandpa, bit_idx, edge_flow, single_child, weight::Float32)
        if !single_child
            c::Float32 = ((edge_flow >> bit_idx) & UInt64(0x1)) * weight # cast for @atomic to be happy
            CUDA.@atomic edge_counts_device[element] += c
        end
    end
    @inline function on_edge(flows, values, prime, sub, element, grandpa, sample_idx, 
                             edge_flow::AbstractFloat, single_child, weight::Nothing)
        if !single_child
            c::Float32 = edge_flow # cast for @atomic to be happy
            CUDA.@atomic edge_counts_device[element] += c
        end
    end
    @inline function on_edge(flows, values, prime, sub, element, grandpa, sample_idx, 
                             edge_flow::AbstractFloat, single_child, weight::Float32)
        if !single_child
            c::Float32 = edge_flow * weight # cast for @atomic to be happy
            CUDA.@atomic edge_counts_device[element] += c
        end
    end
    
    if isbatched(data)
        v, f = nothing, nothing
        if weights != nothing
            map(zip(data, weights)) do (d, w)
                if w != nothing
                    w = to_gpu(w)
                end
                v, f = satisfies_flows(to_gpu(bc), to_gpu(d), v, f; on_node = on_node, on_edge = on_edge, weights = w)
            end
        else
            map(data) do d
                v, f = satisfies_flows(to_gpu(bc), to_gpu(d), v, f; on_node = on_node, on_edge = on_edge, weights = nothing)
            end
        end
    else
        if weights != nothing
            weights = to_gpu(weights)
        end
        v, f = satisfies_flows(to_gpu(bc), to_gpu(data); on_node = on_node, on_edge = on_edge, weights = weights)
    end
    
    CUDA.unsafe_free!(v) # save the GC some effort
    CUDA.unsafe_free!(f) # save the GC some effort

    # TODO: reinstate simpler implementation once https://github.com/JuliaGPU/GPUArrays.jl/issues/313 is fixed and released
    @inbounds parents = bc.elements[1,:]
    @inbounds parent_counts = node_counts[parents]
    @inbounds parent_elcount = bc.nodes[2,parents] .- bc.nodes[1,parents] .+ 1 
    params = log.((edge_counts .+ (pseudocount ./ parent_elcount)) 
                    ./ (parent_counts .+ pseudocount))
    
    to_cpu(params)
end

"""
Uniform distribution
"""
function uniform_parameters(pc::ProbCircuit; perturbation::Float64 = 0.0)
    foreach(pc) do pn
        if is⋁gate(pn)
            if num_children(pn) == 1
                pn.log_probs .= 0.0
            else
                if perturbation < 1e-8
                    pn.log_probs .= log.(ones(Float64, num_children(pn)) ./ num_children(pn))
                else
                    unnormalized_probs = map(x -> 1.0 - perturbation + x * 2 * perturbation, rand(num_children(pn)))
                    pn.log_probs .= log.(unnormalized_probs ./ sum(unnormalized_probs))
                end
            end
        end
    end
end

"""
Expectation maximization parameter learning given missing data
"""
function estimate_parameters_em(pc::ProbCircuit, data; pseudocount::Float64, 
                                use_sample_weights::Bool = true, use_gpu::Bool = false,
                                exp_update_factor = 0.0, update_per_batch::Bool = false)
    if update_per_batch && isbatched(data)
        estimate_parameters_em_per_batch(pc, data; pseudocount, use_sample_weights, use_gpu, 
                                         exp_update_factor)
    end
    
    if isweighted(data)
        # `data' is weighted according to its `weight' column
        data, weights = split_sample_weights(data)
    else
        use_sample_weights = false
    end
    
    pbc = ParamBitCircuit(pc, data; reset=false)
    if isgpu(data)
        use_gpu = true
    elseif use_gpu && !isgpu(data)
        data = to_gpu(data)
    end
    
    params = if use_gpu
        if !isgpu(data)
            data = to_gpu(data)
        end
        if use_sample_weights
            estimate_parameters_gpu(to_gpu(pbc), data, pseudocount; weights)
        else
            estimate_parameters_gpu(to_gpu(pbc), data, pseudocount)
        end
    else
        if use_sample_weights
            estimate_parameters_cpu(pbc, data, pseudocount; weights)
        else
            estimate_parameters_cpu(pbc, data, pseudocount)
        end
    end
    
    estimate_parameters_cached!(pc, pbc.bitcircuit, params; exp_update_factor)
    
    params
end
function estimate_parameters_em_per_batch(pc::ProbCircuit, data; pseudocount::Float64, 
                                          use_sample_weights::Bool = true, use_gpu::Bool = false,
                                          exp_update_factor = 0.0)
    if isgpu(data)
        use_gpu = true
    elseif use_gpu && !isgpu(data)
        data = to_gpu(data)
    end
    
    pbc = ParamBitCircuit(pc, data; reset=false)
    if use_gpu
        pbc = to_gpu(pbc)
    end
    
    reuse_v, reuse_f = nothing, nothing
    reuse_counts = use_gpu ? (nothing, nothing) : (nothing, nothing, nothing, nothing, nothing)
    
    for idx = 1 : length(data)
        pbc, reuse_v, reuse_f, reuse_counts = estimate_parameters_em(pbc, data[idx]; pseudocount, use_gpu, reuse_v, reuse_f, reuse_counts, exp_update_factor)
    end
    
    estimate_parameters_cached!(pc, pbc)
end
function estimate_parameters_em(pbc::ParamBitCircuit, data; pseudocount::Float64, 
                                use_sample_weights::Bool = true, use_gpu::Bool = false,
                                reuse_v = nothing, reuse_f = nothing, reuse_counts = nothing,
                                exp_update_factor = 0.0)
    if isweighted(data)
        # `data' is weighted according to its `weight' column
        data, weights = split_sample_weights(data)
    else
        use_sample_weights = false
    end
    
    if isgpu(data)
        use_gpu = true
    elseif use_gpu && !isgpu(data)
        data = to_gpu(data)
    end
    
    if reuse_counts === nothing
        reuse_counts = use_gpu ? (nothing, nothing) : (nothing, nothing, nothing, nothing, nothing)
    end
    
    params, v, f, reuse_counts = if use_gpu
        if !isgpu(pbc)
            pbc.bitcircuit = to_gpu(pbc.bitcircuit)
            pbc.params = to_gpu(pbc.params)
        end
        if use_sample_weights
            estimate_parameters_gpu(pbc, data, pseudocount; weights, reuse = true, reuse_v, reuse_f, reuse_counts)
        else
            estimate_parameters_gpu(pbc, data, pseudocount; reuse = true, reuse_v, reuse_f, reuse_counts)
        end
    else
        if use_sample_weights
            estimate_parameters_cpu(pbc, data, pseudocount; weights, reuse = true, reuse_v, reuse_f, reuse_counts)
        else
            estimate_parameters_cpu(pbc, data, pseudocount; reuse = true, reuse_v, reuse_f, reuse_counts)
        end
    end
    
    # Update the parameters to `pbc`
    if use_gpu # GPU
        tempparam = Vector{Float64}(undef, length(params))
        tempparam .= to_cpu(params)
        @inbounds @views pbc.params .+= log(exp_update_factor)
        @inbounds @views params .+= log(1.0 - exp_update_factor)
        delta = @inbounds @views @. CUDA.ifelse(pbc.params == params, CUDA.zero(params), CUDA.abs(pbc.params - params))
        @inbounds @views @. pbc.params = CUDA.max(pbc.params, params) + CUDA.log1p(CUDA.exp(-delta))
        
        CUDA.unsafe_free!(params)
        CUDA.unsafe_free!(delta)
    else # CPU
        @inbounds @views pbc.params = logaddexp.(pbc.params .+ log(exp_update_factor), params .+ log(1.0 - exp_update_factor))
    end
    
    pbc, v, f, reuse_counts
end

function estimate_parameters_cpu(pbc::ParamBitCircuit, data, pseudocount; weights = nothing, reuse::Bool = false,
                                 reuse_v = nothing, reuse_f = nothing, 
                                 reuse_counts = (nothing, nothing, nothing, nothing, nothing))
    # `data` is batched?
    data_batched = isbatched(data)
    
    # no need to synchronize, since each computation is unique to a decision node
    node_counts::Vector{Float64} = similar!(reuse_counts[1], Vector{Float64}, num_nodes(pbc.bitcircuit))
    @inbounds @views @. @avx node_counts .= typemin(eltype(node_counts)) # log(0)
    
    # rescale pseudocount using the average weight of samples
    if weights !== nothing
        if data_batched
            pseudocount = pseudocount * mapreduce(sum, +, weights) / num_examples(data)
        else
            pseudocount = pseudocount * sum(weights) / size(weights, 1)
        end
    end
    
    bc = pbc.bitcircuit
    params = pbc.params
    
    edge_counts::Vector{Float64} = similar!(reuse_counts[2], Vector{Float64}, num_elements(bc))
    parent_node_counts::Vector{Float64} = similar!(reuse_counts[3], Vector{Float64}, num_elements(bc))
    @inbounds @views edge_counts[:] .= zero(Float64)
    @inbounds @views parent_node_counts[:] .= zero(Float64)
    
    # Buffer to save some allocations
    buffer::Vector{Float64} = similar!(reuse_counts[4], Vector{Float64}, 
        data_batched ? num_examples(data[1]) : num_examples(data))
    
    # Pre-compute log(weights) to save computation time
    if weights !== nothing
        log_weights = similar!(reuse_counts[5], Vector{Float64}, 
            data_batched ? num_examples(data[1]) : num_examples(data))
        
        if !data_batched
            @inbounds @views @avx log_weights .= log.(weights)
        end
    else
        log_weights = nothing
    end
    
    # For batched dataset, we want to enable 'estimate' for parent_node_counts only in the last minibatch
    estimate_flag = true

    @inline function on_node(flows, values, dec_id, weights::Nothing)
        @inbounds @views @. @avx buffer = flows[:, dec_id]
        node_counts[dec_id] = logaddexp(node_counts[dec_id], logsumexp(buffer))
    end
    @inline function on_node(flows, values, dec_id, weights)
        @inbounds @views @. @avx buffer = flows[:, dec_id] + log_weights[:]
        node_counts[dec_id] = logaddexp(node_counts[dec_id], logsumexp(buffer))
    end

    @inline function estimate(element, decision, edge_count)
        edge_counts[element] += exp(edge_count)
        if estimate_flag # For batched dataset, we only accumulate parent_node_counts after all node_counts have been cumulated
            parent_node_counts[element] += exp(node_counts[decision])
        end
    end

    @inline function on_edge(flows, values, prime, sub, element, grandpa, single_child, weights::Nothing)
        θ = eltype(flows)(params[element])
        if !single_child
            @inbounds @views @. @avx buffer = values[:, prime] + values[:, sub] - values[:, grandpa] + flows[:, grandpa] + θ
            buffer .= ifelse.(isnan.(buffer[:]), typemin(eltype(flows)), buffer)
            
            edge_count = logsumexp(buffer)
            
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end
    @inline function on_edge(flows, values, prime, sub, element, grandpa, single_child, weights)
        θ = eltype(flows)(params[element])
        if !single_child
            @inbounds @views @. @avx buffer = values[:, prime] + values[:, sub] - values[:, grandpa] + flows[:, grandpa] + θ + log_weights
            buffer .= ifelse.(isnan.(buffer), typemin(eltype(flows)), buffer)
            
            edge_count = logsumexp(buffer)
            
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end

    if data_batched
        if weights !== nothing
            v, f = reuse_v, reuse_f
            for idx = 1 : length(data)
                d = data[idx]
                w = weights[idx]
                
                # Resize buffer if the current minibatch has a different size
                if size(buffer, 1) != num_examples(d)
                    resize!(buffer, num_examples(d))
                end
                
                # Resize log_weights if the current minibatch has a different size
                if data_batched
                    if size(log_weights, 1) != num_examples(d)
                        resize!(log_weights, num_examples(d))
                    end
                    @inbounds @views @avx log_weights .= log.(w)
                end
                
                # Only accumulate parent_node_counts in the final call to marginal_flows
                if idx == length(data)
                    estimate_flag = true
                else
                    estimate_flag = false
                end

                v, f = marginal_flows(pbc, d, v, f; on_node = on_node, on_edge = on_edge, weights = w)
            end
        else
            v, f = reuse_v, reuse_f
            for idx = 1 : length(data)
                d = data[idx]
                
                # Resize buffer if the current minibatch has a different size
                if size(buffer, 1) != num_examples(d)
                    resize!(buffer, num_examples(d))
                end
                
                if idx == length(data)
                    estimate_flag = true
                else
                    estimate_flag = false
                end
                
                v, f = marginal_flows(pbc, d, v, f; on_node = on_node, on_edge = on_edge, weights = nothing)
            end
        end
    else
        v, f = marginal_flows(pbc, data, reuse_v, reuse_f; on_node, on_edge, weights)
    end
    
    # `edge_counts` now becomes "params"
    @simd for i = 1 : num_elements(bc)
        num_els = num_elements(bc.nodes, bc.elements[1, i])
        if num_els == 1
            @inbounds edge_counts[i] = zero(eltype(edge_counts)) # log(1)
        else
            @inbounds edge_counts[i] = log((edge_counts[i] + pseudocount / num_elements(bc.nodes, bc.elements[1, i])) / (parent_node_counts[i] + pseudocount))
        end
    end

    if reuse
        # Also return the allocated vars v, f, counts for future reuse
        edge_counts, v, f, (node_counts, edge_counts, parent_node_counts, buffer, log_weights)
    else
        edge_counts # a.k.a. log_probs
    end
end

function estimate_parameters_gpu(pbc::ParamBitCircuit, data, pseudocount; weights = nothing, reuse::Bool = false,
                                 reuse_v = nothing, reuse_f = nothing, reuse_counts = (nothing, nothing))
    # `data` is batched?
    data_batched = isbatched(data)
    
    # rescale pseudocount using the average weight of samples
    if weights !== nothing
        if data_batched
            if isgpu(weights)
                pseudocount = pseudocount * mapreduce(sum, +, to_cpu(weights)) / num_examples(data)
            else
                pseudocount = pseudocount * mapreduce(sum, +, weights) / num_examples(data)
            end
        else
            pseudocount = pseudocount * sum(weights) / size(weights, 1)
        end
    end
    
    bc = pbc.bitcircuit
    node_counts::CuVector{Float64} = similar!(reuse_counts[1], CuVector{Float64}, num_nodes(bc))
    edge_counts::CuVector{Float64} = similar!(reuse_counts[2], CuVector{Float64}, num_elements(bc))
    @inbounds @views node_counts[:] .= zero(Float64)
    @inbounds @views edge_counts[:] .= zero(Float64)
    # need to manually cudaconvert closure variables
    node_counts_device = CUDA.cudaconvert(node_counts)
    edge_counts_device = CUDA.cudaconvert(edge_counts)
        
    @inline function on_node(flows, values, dec_id, chunk_id, flow, weight::Nothing)
        c::Float64 = exp(flow) # cast for @atomic to be happy
        CUDA.@atomic node_counts_device[dec_id] += c
    end
    @inline function on_node(flows, values, dec_id, chunk_id, flow, weight::AbstractFloat)
        c::Float64 = exp(flow) * weight # cast for @atomic to be happy
        CUDA.@atomic node_counts_device[dec_id] += c
    end

    @inline function on_edge(flows, values, prime, sub, element, grandpa, chunk_id, edge_flow, single_child, weight::Nothing)
        if !single_child
            c::Float64 = exp(edge_flow) # cast for @atomic to be happy
            CUDA.@atomic edge_counts_device[element] += c
        end
    end
    @inline function on_edge(flows, values, prime, sub, element, grandpa, chunk_id, edge_flow, single_child, weight::AbstractFloat)
        if !single_child
            c::Float64 = exp(edge_flow) * weight # cast for @atomic to be happy
            CUDA.@atomic edge_counts_device[element] += c
        end
    end

    if data_batched
        v, f = reuse_v, reuse_f
        if weights != nothing
            map(zip(data, weights)) do (d, w)
                if w != nothing
                    w = to_gpu(w)
                end
                v, f = marginal_flows(pbc, d, v, f; on_node = on_node, on_edge = on_edge, weights = w)
                
                nothing # Return nothing to save some time
            end
        else
            map(data) do d
                v, f = marginal_flows(pbc, d, v, f; on_node = on_node, on_edge = on_edge, weights = nothing)
                
                nothing # Return nothing to save some time
            end
            
            nothing # Return nothing to save some time
        end
    else
        if weights != nothing
            weights = to_gpu(weights)
        end
        
        v, f = marginal_flows(pbc, data, reuse_v, reuse_f; on_node, on_edge, weights)
    end

    # TODO: reinstate simpler implementation once https://github.com/JuliaGPU/GPUArrays.jl/issues/313 is fixed and released
    @inbounds parents = bc.elements[1,:]
    @inbounds parent_counts = node_counts[parents]
    @inbounds parent_elcount = bc.nodes[2,parents] .- bc.nodes[1,parents] .+ 1 
    params = log.((edge_counts .+ (pseudocount ./ parent_elcount)) 
                    ./ (parent_counts .+ pseudocount))
    params = @inbounds @views @. ifelse(parent_elcount == 1, zero(params), params)
    
    CUDA.unsafe_free!(parents)
    CUDA.unsafe_free!(parent_counts)
    CUDA.unsafe_free!(parent_elcount)
    
    # Only free the memory if the reuse memory is not provided
    if !reuse
        CUDA.unsafe_free!(v) # save the GC some effort
        CUDA.unsafe_free!(f) # save the GC some effort
        CUDA.unsafe_free!(node_counts) # save the GC some effort
        CUDA.unsafe_free!(edge_counts) # save the GC some effort
    end
    
    if reuse
        # Also return the allocated vars v, f, counts for future reuse
        params, v, f, (node_counts, edge_counts)
    else
        to_cpu(params) # a.k.a. log_probs
    end
end

"""
    apply_entropy_reg(pc::ProbCircuit; alpha::Float64)::Nothing

Add entropy regularization to a deterministic (see LogicCircuits.isdeterministic) probabilistic
circuit. `alpha` is a hyperparameter that balances the weights between the likelihood and the 
entropy: \argmax_{\theta} L(\theta) = ll_mean(\theta) + alpha * entropy(\theta).
"""
function apply_entropy_reg(pc::ProbCircuit; alpha::Float64)::Nothing
    p = Vector{Float64}(undef, 0) # For reuse
    p_exp_logprob = Vector{Float64}(undef, 0) # For reuse
    
    update_params(n::ProbCircuit) = begin
        if is⋁gate(n) && length(children(n)) > 1
            resize!(p, length(n.log_probs))
            resize!(p_exp_logprob, length(n.log_probs))
            @inbounds @views p .= exp.(n.log_probs)
            for _ = 1 : 3
                y = sum(alpha .* n.log_probs .- p .* exp.(-n.log_probs)) / length(n.log_probs)
                for _ = 1 : 4
                    @inbounds @views p_exp_logprob .= p .* exp.(-n.log_probs)
                    @inbounds @views n.log_probs .+= (p_exp_logprob .- alpha .* n.log_probs .+ y) ./ (p_exp_logprob .+ alpha)
                    @inbounds @views n.log_probs .-= logsumexp(n.log_probs) # Project the parameters back to its valid space
                end
            end
        end
    end
    
    foreach_down(update_params, pc)
end