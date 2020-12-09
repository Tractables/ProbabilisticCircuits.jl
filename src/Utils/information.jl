export pairwise_marginals, mutual_information, set_mutual_information

using Statistics
using StatsFuns: xlogx, xlogy
using LogicCircuits: issomething
using CUDA: CUDA, CuMatrix, CuVector, CuArray


"Cache pairwise / marginal distribution for all variables in one dataset"
mutable struct DisCache
    pairwise::Array{Float64}
    marginal::Array{Float64}
end

@inline dimension(discache::DisCache) = size(discache.marginal)[1]
DisCache(num) = DisCache(Array{Float64}(undef, num, num, 4), Array{Float64}(undef, num, 2))

#####################
# Methods for pairwise and marginal distribution
#####################

#TODO: give a better name; make this the default `pairwise_marginals` for all binary data on CPU?
function cache_distributions(bm, w::Union{Nothing, Vector}=nothing; α, flag=(pairwise=true, marginal=true))
    
    # parameters
    D = size(bm)[2]
    N = issomething(w) ? sum(w) : size(bm)[1]
    m = convert(Matrix{Float64}, bm)
    notm = convert(Matrix{Float64}, .!bm)

    dis_cache = DisCache(D)
    base = N + 4 * α
    w = isnothing(w) ? ones(Float64, N) : w

    # pairwise distribution
    if flag.pairwise
        dis_cache.pairwise[:,:,1] = (notm' * (notm .* w) .+ α) / base   # p00
        dis_cache.pairwise[:,:,2] = (notm' * (m .* w) .+ α) / base      # p01
        dis_cache.pairwise[:,:,3] = (m' * (notm .* w) .+ α) / base      # p10
        dis_cache.pairwise[:,:,4] = (m' * (m .* w) .+ α) / base         # p11
    end
    # marginal distribution

    if flag.marginal
        dis_cache.marginal[:, 1] = (sum(notm .* w, dims=1) .+ 2 * α) / base
        dis_cache.marginal[:, 2] = (sum(m .* w, dims=1).+ 2 * α) / base
    end
    dis_cache
end

"Compute an array giving all pairwise marginals estimated on empirical (weighted) data"
function pairwise_marginals(data::Matrix, weights::Vector, num_cats = maximum(data); pseudocount = 1.0)
    
    @assert minimum(data) > 0 "Categorical data labels are assumed to be indexed starting 1"
    num_vars = size(data,2)
    pair_margs = Array{Float32}(undef, num_vars, num_vars, num_cats, num_cats)
    Z = sum(weights) + pseudocount
    single_smooth = pseudocount / num_cats
    pair_smooth = single_smooth / num_cats
    
    for i = 1:num_vars 
        Threads.@threads for j = 1:num_vars # note: multithreading needs to be on inner loop for thread-safe copying across diagonal
            if i<=j
                if i!=j
                    @inbounds pair_margs[i,j,:,:] .= pair_smooth
                else
                    @inbounds pair_margs[i,j,:,:] .= zero(eltype(pair_margs))
                    for l = 1:num_cats
                        @inbounds pair_margs[i,j,l,l] = single_smooth
                    end
                end
                @simd for k = 1:size(data,1) # @avx here gives incorrect results
                    @inbounds pair_margs[i,j,data[k,i],data[k,j]] += weights[k]
                end
                @inbounds pair_margs[i,j,:,:] ./= Z
            else
                @inbounds pair_margs[i,j,:,:] .= (@view pair_margs[j,i,:,:])' 
            end
            nothing
        end
    end

    return pair_margs
end

function pairwise_marginals(data::CuMatrix, weights::CuVector, num_cats = maximum(data); pseudocount = 1.0)
    
    @assert minimum(data) > 0 "Categorical data labels are assumed to be indexed starting 1"
    num_vars = size(data,2)
    pair_margs = CuArray{Float32}(undef, num_vars, num_vars, num_cats, num_cats)
    Z = sum(weights) + pseudocount
    single_smooth = pseudocount / num_cats
    pair_smooth = single_smooth / num_cats
    
    data_device = CUDA.cudaconvert(data)
    weights_device = CUDA.cudaconvert(weights)
    pair_margs_device = CUDA.cudaconvert(pair_margs)

    var_indices = CuArray(1:num_vars)
    CUDA.@sync begin
        broadcast(var_indices, var_indices') do i,j
            if i <= j
                if i!=j
                    @inbounds pair_margs_device[i,j,:,:] .= pair_smooth
                else
                    @inbounds pair_margs_device[i,j,:,:] .= zero(Float32)
                    for l = 1:num_cats
                        @inbounds pair_margs_device[i,j,l,l] = single_smooth
                    end
                end
                for k = 1:size(data_device,1)
                    pair_margs_device[i,j,data_device[k,i],data_device[k,j]] += weights_device[k]
                end
            end
            nothing
        end
        pair_margs ./= Z
        broadcast(var_indices, var_indices') do i,j
            if i > j
                for l = 1:num_cats, m = 1:num_cats
                    @inbounds pair_margs_device[i,j,l,m] = pair_margs_device[j,i,m,l] 
                end
            end
            nothing
        end
    end
    return pair_margs
end

#####################
# Mutual Information
#####################

function mutual_information(dis_cache::DisCache)
    D = dimension(dis_cache)
    p0 = @view dis_cache.marginal[:, 1]
    p1 = @view dis_cache.marginal[:, 2]
    pxpy = Array{Float64}(undef, D, D, 4)
    pxpy[:,:,1] = p0 * p0'
    pxpy[:,:,2] = p0 * p1'
    pxpy[:,:,3] = p1 * p0'
    pxpy[:,:,4] = p1 * p1'
    pxy_log_pxy = @. xlogx(dis_cache.pairwise)
    pxy_log_pxpy = @. xlogy(dis_cache.pairwise, pxpy)
    dropdims(sum(pxy_log_pxy - pxy_log_pxpy,dims=3), dims=3)
end

"Calculate mutual information of given bit matrix `bm`, example weights `w`, and smoothing pseudocount `α`"
function mutual_information(bm, w::Union{Nothing, Vector}=nothing; α)
    dis_cache = cache_distributions(bm, w; α=α)
    mi = mutual_information(dis_cache)
    return (dis_cache, mi)
end

"Calculate set mutual information"
function set_mutual_information(mi::Matrix, sets::AbstractVector{<:AbstractVector})::Matrix
    len = length(sets)
    if len == size(mi)[1]
        return mi
    end

    pmi = zeros(len, len)
    for i in 1 : len, j in i + 1 : len
        pmi[i, j] = pmi[j, i] = mean(mi[sets[i], sets[j]])
    end
    return pmi
end