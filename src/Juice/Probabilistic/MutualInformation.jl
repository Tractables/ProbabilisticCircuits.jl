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

"Calculate mutual information of given data set"
function mutual_information(train_x::WXData; α = 0)
    # get parameters
    D = num_features(train_x)
    w = Data.weights(train_x)
    N = sum(w)
    m = @. Float64($feature_matrix(train_x))
    notm = @. Float64(!$feature_matrix(train_x))
    dis_cache = DisCache(D)

    # pairwise distribution
    base = N + 4 * α
    dis_cache.pairwise[:,:,1] = p00 = (notm' * (notm .* w) .+ α) / base
    dis_cache.pairwise[:,:,2] = p01 = (notm' * (m .* w) .+ α) / base
    dis_cache.pairwise[:,:,3] = p10 = (m' * (notm .* w) .+ α) / base
    dis_cache.pairwise[:,:,4] = p11 = (m' * (m .* w) .+ α) / base

    # marginal distribution
    dis_cache.marginal[:, 1] = p0 = (sum(notm .* w, dims=1) .+ 2 * α) / base
    dis_cache.marginal[:, 2] = p1 = (sum(m .* w, dims=1).+ 2 * α) / base

    # mutual information
    pxpy = Array{Float64}(undef, D, D, 4)
    pxpy[:,:,1] = p0' * p0
    pxpy[:,:,2] = p0' * p1
    pxpy[:,:,3] = p1' * p0
    pxpy[:,:,4] = p1' * p1
    pxy_log_pxy = @. xlogx(dis_cache.pairwise)
    pxy_log_pxpy = @. xlogy(dis_cache.pairwise, pxpy)
    mi = sum(pxy_log_pxy - pxy_log_pxpy,dims=3)[:,:,1]

    return (dis_cache, mi)
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