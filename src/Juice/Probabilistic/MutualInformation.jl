using Statistics
using LinearAlgebra
using StatsFuns

"Cache pairwise / marginal distribution for all variables in one dataset"
mutable struct DisCache
    pairwise::Array{Float64}
    marginal::Array{Float64}
end

DisCache(num) = DisCache(Array{Float64}(undef, num, num, 4), Array{Float64}(undef, num, 2))

#####################
# Methods for pairwise and marginal distribution
#####################
@inline get_parameters(bm::BitMatrix, α) = size(bm)[2], Float64(size(bm)[1]), @. Float64(bm), @. Float64(!bm)
@inline get_parameters(bm::BitMatrix, w, α) = size(bm)[2], sum(w), @. Float64(bm), Float64(!bm)

function get_mutual_information(dis_cache::DisCache, D, p0, p1)
    pxpy = Array{Float64}(undef, D, D, 4)
    pxpy[:,:,1] = p0' * p0
    pxpy[:,:,2] = p0' * p1
    pxpy[:,:,3] = p1' * p0
    pxpy[:,:,4] = p1' * p1
    pxy_log_pxy = @. xlogx(dis_cache.pairwise)
    pxy_log_pxpy = @. xlogy(dis_cache.pairwise, pxpy)
    dropdims(sum(pxy_log_pxy - pxy_log_pxpy,dims=3), dims=3)
end

"Calculate mutual information of given bit matrix `bm`, and smoothing pseudocount `α`"
# speed up for single model / unweighted data, save 4 times matrix .* vector
function mutual_information(bm::BitMatrix; α) 
    # get parameters
    D, N, (m, notm) = get_parameters(bm, α)
    dis_cache = DisCache(D)
    base = N + 4 * α

    # pairwise distribution
    dis_cache.pairwise[:,:,1] = (notm' * notm .+ α) / base  # p00
    dis_cache.pairwise[:,:,2] = (notm' * m .+ α) / base     # p01
    dis_cache.pairwise[:,:,3] = (m' * notm .+ α) / base     # p10
    dis_cache.pairwise[:,:,4] = (m' * m .+ α) / base        # p11

    # marginal distribution
    dis_cache.marginal[:, 1] = p0 = (sum(notm, dims=1) .+ 2 * α) / base
    dis_cache.marginal[:, 2] = p1 = (sum(m, dims=1).+ 2 * α) / base

    # mutual information
    mi = get_mutual_information(dis_cache, D, p0, p1)
    return (dis_cache, mi)
end

"Calculate mutual information of given bit matrix `bm`, example weights `w`, and smoothing pseudocount `α`"
function mutual_information(bm::BitMatrix, w::Vector{Float64}; α)
    # get parameters
    D, N, (m, notm) = get_parameters(bm, w, α)
    dis_cache = DisCache(D)
    base = N + 4 * α

    # pairwise distribution
    dis_cache.pairwise[:,:,1] = (notm' * (notm .* w) .+ α) / base
    dis_cache.pairwise[:,:,2] = (notm' * (m .* w) .+ α) / base
    dis_cache.pairwise[:,:,3] = (m' * (notm .* w) .+ α) / base
    dis_cache.pairwise[:,:,4] = (m' * (m .* w) .+ α) / base

    # marginal distribution
    dis_cache.marginal[:, 1] = p0 = (sum(notm .* w, dims=1) .+ 2 * α) / base
    dis_cache.marginal[:, 2] = p1 = (sum(m .* w, dims=1).+ 2 * α) / base

    # mutual information
    mi = get_mutual_information(dis_cache, D, p0, p1)
    return (dis_cache, mi)
end

mutual_information(train_x::PlainXData; α) = mutual_information(feature_matrix(train_x); α=α)
mutual_information(train_x::PlainXData, w::Vector{Float64}; α) = mutual_information(feature_matrix(train_x), w; α=α)
mutual_information(train_x::WXData; α) = mutual_information(feature_matrix(train_x), Data.weights(train_x); α=α)
# mutual_information(train_x::XBatches{<:Bool,<:PlainXData{<:Bool}}, w::Vector{Float64}; α) = mutual_information(unbatch(train_x), w; α=α)

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