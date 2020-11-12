export mutual_information, set_mutual_information
using Statistics
using StatsFuns: xlogx, xlogy
using LogicCircuits: issomething

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