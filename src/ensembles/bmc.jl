export BayesModelComb, bmc_sample_psdd

using Distributions: Dirichlet

"Bayesian Model Combination."
mutable struct BayesModelComb{T <: ProbCircuit}
    E::Vector{Ensemble{T}}
    W::Vector{Float64}
end

"Constructs a SamplePSDD BMC with `q*t` combinations, each with `n` models."
function bmc_sample_psdd(n::Integer, ϕ::Diagram, k::Integer, D::DataFrame, q::Integer, t::Integer;
        reuse::Union{Vector{StructProbCircuit}, Nothing} = nothing, vtree_bias::Real = -1.0,
        α::Union{Vector{Float64}, Nothing} = nothing, verbose::Bool = true, kwargs...)::BayesModelComb{StructProbCircuit}
    if isnothing(α) α = ones(n) end
    K = q*t
    M = K*n
    v = ncol(D)
    if isnothing(reuse)
        circs = Vector{StructProbCircuit}(undef, M)
        verbose && print("Sampling ", M, " PSDDs...\n  ")
        @qthreads for i ∈ 1:M
            circs[i] = sample_psdd(ϕ, sample_vtree(v, vtree_bias), k, D; kwargs...)
            verbose && print('*')
        end
        verbose && print('\n')
    else circs = reuse end
    E = Vector{Ensemble{StructProbCircuit}}(undef, K)
    dirichlet = Dirichlet(α)
    LL = Vector{Float64}(undef, K)
    W = Vector{Vector{Float64}}(undef, q)
    e = 1
    for i ∈ 1:t
        i_max, max_ll = -1, -Inf
        for j ∈ 1:q
            W[j] = rand(dirichlet)
            E[e] = Ensemble{StructProbCircuit}(circs[(e-1)*n+1:e*n], log.(W[j]))
            ll = sum(log_likelihood_per_instance(E[e], D))
            # Assume a uniform prior on the ensembles so that max p(e|D) = max p(D|e).
            if ll > max_ll i_max, max_ll = j, ll end
            LL[e] = ll
            verbose && println("BMC Iteration ", e, '/', K, '.')
            e += 1
        end
        α .+= W[i_max]
    end
    LL .= exp.(LL .- maximum(LL))
    LL .= LL ./ sum(LL)
    return BayesModelComb(E, log.(LL))
end

function weighted_query(B::BayesModelComb{T}, D::DataFrame, f::Function; kwargs...)::Vector{Float64} where T <: ProbCircuit
    n, m = nrow(D), length(B.E)
    LL = Matrix{Float64}(undef, n, m)
    @inbounds for i ∈ 1:m LL[:,i] .= f(B.E[i], D; kwargs...) .+ B.W[i] end
    return logsumexp(LL, 2)
end

@inline function log_likelihood_per_instance(B::BayesModelComb{T}, D::DataFrame; use_gpu::Bool = false)::Vector{Float64} where T <: ProbCircuit
    return weighted_query(B, D, log_likelihood_per_instance; use_gpu)
end

@inline function marginal(B::BayesModelComb{T}, D::DataFrame)::Vector{Float64} where T <: ProbCircuit
    return weighted_query(B, D, marginal)
end
