export entropy, conditional_entropy, mutual_information
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
@inline get_parameters(bm, α, w=nothing) = size(bm)[2], issomething(w) ? sum(w) : size(bm)[1], convert(Matrix{Float64}, bm), convert(Matrix{Float64}, .!bm)

function cache_distributions(bm, w::Union{Nothing, Vector}=nothing; α, flag=(pairwise=true, marginal=true))
    # parameters
    D, N, m, notm = get_parameters(bm, α, w)
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
function set_mutual_information(mi::Matrix, sets::Vector{Vector})::Matrix
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
function entropy(dis_cache::DisCache)
    D = dimension(dis_cache)
    px_log_px = @. xlogx(dis_cache.marginal)
    - dropdims(sum(px_log_px; dims=2); dims=2)
end

function entropy(bm::AbstractMatrix{<:Bool}, w::Union{Nothing, AbstractVector{<:AbstractFloat}}=nothing; α)
    dis_cache = cache_distributions(bm, w; α=α, flag=(pairwise=true, marginal=true))
    return (dis_cache, entropy(dis_cache))
end

function sum_entropy_given_x(bm::AbstractMatrix{<:Bool}, x, w::Union{Nothing, AbstractVector{<:AbstractFloat}}=nothing; α)::Float64
    @assert x <= size(bm)[2]
    vars = [1 : x-1; x+1 : size(bm)[2]]
    indexes_left = bm[:,x].== 0
    indexes_right = bm[:,x] .== 1
    w1 = sum(Float64.(indexes_left))
    w2 = sum(Float64.(indexes_right))
    w1, w2 = w1 / (w1 + w2), w2 / (w1 + w2)
    subm_left = @view bm[indexes_left, vars]
    subm_right = @view bm[indexes_right, vars]
    
    w_left = issomething(w) ? (@view w[indexes_left]) : nothing
    w_right = issomething(w) ? (@view w[indexes_right]) : nothing

    w1 * sum(entropy(subm_left, w_left; α=α)[2]) + w2 * sum(entropy(subm_right, w_right; α=α)[2])
end

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
