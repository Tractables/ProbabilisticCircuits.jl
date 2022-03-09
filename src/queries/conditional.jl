"""
    conditional_loglikelihoods(bpc::CuBitsProbCircuit, data_q::CuArray, data_evi::CuArray; batch_size, mars_mem = nothing)

Returns conditional_loglikelihoods for each datapoint on gpu. Missing values should be denoted by `missing`.
- `bpc`: BitCircuit on gpu
- `data_q`: CuArray{Union{Missing, data_types...}}
- `data_evi`: CuArray{Union{Missing, data_types...}}
- `batch_size`
- `mars_mem`: Not required, advanced usage. CuMatrix to reuse memory and reduce allocations. See `prep_memory` and `cleanup_memory`.
"""
function conditional_loglikelihoods(bpc::CuBitsProbCircuit, data_q::CuArray, data_evi::CuArray; 
                        batch_size, mars_mem = nothing, 
                        mine=2, maxe=32, debug=false)

    @assert size(data_q, 1) == 1 "For now supporting only one input in data_q $(size(data_q))"
    num_nodes = length(bpc.nodes)

    # 0. Find conflicts
    # data_q and data_evi share different instantiations to one x_i
    # if there is conflict the log p(q,e) = -Inf32
    cond_lls = CUDA.zeros(Float32)(size(data_evi, 1))
    conflicts = any(.!(ismissing.(data_q) .| ismissing.(data_evi) .| isequal.(data_q, data_evi)), dims=2)
    
    # 1. Compute P(data_evi)
    data = CuArray{Union{Missing, eltype(data_evi)}}(undef, size(data_evi)...)
    CUDA.copy!(data, data_evi)

    marginals = prep_memory(mars_mem, (batch_size, num_nodes), (false, true));
    evi_lls = loglikelihoods(bpc, data; batch_size, marginals);

    # 2. Compute P(data_q, data_evi)
    flag = .!ismissing.(data_q)
    data[:,  flag] = data_q[1, flag]
    q_evi_lls = loglikelihoods(bpc, data; batch_size, marginals);

    cleanup_memory(marginals, mars_mem)
    cond_lls .= q_evi_lls .- evi_lls
    cond_lls[conflicts] = -Inf32

    return cond_lls
end
