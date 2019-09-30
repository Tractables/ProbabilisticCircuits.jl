using Statistics
using LinearAlgebra
using StatsFuns

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
@inline get_parameters(bm::AbstractMatrix{<:Bool}, α, w=nothing) = size(bm)[2], issomething(w) ? sum(w) : size(bm)[1], @. Float64(bm), @. Float64(!bm)

function cache_distributions(bm::AbstractMatrix{<:Bool}, w::Union{Nothing, AbstractVector{<:AbstractFloat}}=nothing; α)
    # parameters
    D, N, (m, notm) = get_parameters(bm, α, w)
    dis_cache = DisCache(D)
    base = N + 4 * α
    w = isnothing(w) ? ones(Float64, N) : w

    # pairwise distribution
    dis_cache.pairwise[:,:,1] = (notm' * (notm .* w) .+ α) / base   # p00
    dis_cache.pairwise[:,:,2] = (notm' * (m .* w) .+ α) / base      # p01
    dis_cache.pairwise[:,:,3] = (m' * (notm .* w) .+ α) / base      # p10
    dis_cache.pairwise[:,:,4] = (m' * (m .* w) .+ α) / base         # p11

    # marginal distribution
    dis_cache.marginal[:, 1] = (sum(notm .* w, dims=1) .+ 2 * α) / base
    dis_cache.marginal[:, 2] = (sum(m .* w, dims=1).+ 2 * α) / base

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
function mutual_information(bm::AbstractMatrix{<:Bool}, w::Union{Nothing, AbstractVector{<:AbstractFloat}}=nothing; α)
    dis_cache = cache_distributions(bm, w; α=α)
    mi = mutual_information(dis_cache)
    return (dis_cache, mi)
end

function mutual_information(train_x::PlainXData, w::Union{Nothing, AbstractVector{<:AbstractFloat}}=nothing; α)
    mutual_information(feature_matrix(train_x), w; α=α)
end

"Calculate set mutual information"
function set_mutual_information(mi::Matrix, sets::Vector{Vector{Var}})::Matrix
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

#####################
# Entropy
#####################
function conditional_entropy(dis_cache::DisCache)
    D = dimension(dis_cache)
    pxy_log_pxy = @. xlogx(dis_cache.pairwise)
    pxy = dis_cache.pairwise
    log_px = log.(dis_cache.marginal)
    
    pxy_log_px = Array{Float64}(undef, D, D, 4)
    pxy_log_px[:, :, 1] = view(pxy, :, :, 1) .* view(log_px, :, 1)
    pxy_log_px[:, :, 2] = view(pxy, :, :, 2) .* view(log_px, :, 1)
    pxy_log_px[:, :, 3] = view(pxy, :, :, 3) .* view(log_px, :, 2)
    pxy_log_px[:, :, 4] = view(pxy, :, :, 4) .* view(log_px, :, 2)
    h_y_given_x = - dropdims(sum(pxy_log_pxy - pxy_log_px; dims=3); dims=3)
end

function conditional_entropy(bm::AbstractMatrix{<:Bool}, w::Union{Nothing, AbstractVector{<:AbstractFloat}}=nothing; α)
    dis_cache = cache_distributions(bm, w; α=α)
    return (dis_cache, conditional_entropy(dis_cache))
end

function set_conditional_entropy(ce::Matrix{<:Float64})
    dropdims(sum(ce; dims=2); dims=2)
end
