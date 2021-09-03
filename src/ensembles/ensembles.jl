export Ensemble, ensemble_sample_psdd, sample_vtree, learn_ensemble_llw!, learn_ensemble_em!,
       learn_ensemble_stacking!

using ..Utils: sample_vtree, kfold
using ThreadPools

"Weighted ensemble of probabilistic circuits."
mutable struct Ensemble{T <: ProbCircuit}
    C::Vector{T}
    W::Vector{Float64}
end

"""
Creates an ensemble of `n` SamplePSDD-generated probabilistic circuits, with `v` the total number
of variables in the data, `ϕ` the logic constraints, `D` the data to be learned and `k` the
maximum number of primes to be sampled.

Keyword arguments for `sample_psdd` are passed down. Optionally, the function takes keyword
argument `vtree_bias`, which samples more left (value closer to 0.0) or right-leaning (value closer
to 1.0) vtrees. If a negative value is given, sample uniformly distributed vtrees.

Weights are computed by the given `strategy`. These can be any one of the following:
    1. `:likelihood` for likelihood weighting;
    2. `:uniform` for uniform weights;
    3. `:em` for Expectation-Maximization;
    4. `:stacking` for mixture model Stacking;
"""
function ensemble_sample_psdd(n::Integer, ϕ::Bdd, k::Int, D::DataFrame; vtree_bias::Real = -1.0,
        strategy::Symbol = :em, verbose::Bool = true, em_maxiter::Integer = 100,
        kfold::Integer = min(nrow(D), 5), pseudocount::Real = 1.0, kwargs...)::Ensemble{StructProbCircuit}
    circs = Vector{StructProbCircuit}(undef, n)
    v = ncol(D)
    verbose && print("Sampling ", n, " PSDDs...\n  ")
    @qthreads for i ∈ 1:n
        V = sample_vtree(v, vtree_bias)
        circs[i] = sample_psdd(ϕ, V, k, D; pseudocount, kwargs...)
        verbose && print('*')
    end
    verbose && print('\n')
    c = log(n)
    E = Ensemble{StructProbCircuit}(circs, -fill(c, n))
    if strategy == :likelihood return learn_ensemble_llw!(E, D)
    elseif strategy == :em return learn_ensemble_em!(E, D; verbose, maxiter = em_maxiter)
    elseif strategy == :stacking
        return learn_ensemble_stacking!(E, D; verbose, maxiter = em_maxiter, k = kfold, pseudocount)
    end
    @assert strategy == :uniform "Unrecognized ensemble strategy."
    return E
end

"Learns the weights of the Ensemble by the likelihood value of data `D`."
function learn_ensemble_llw!(E::Ensemble{T}, D::DataFrame)::Ensemble{T} where T <: ProbCircuit
    n = length(E.C)
    LL = Vector{Float64}(undef, n)
    @qthreads for i ∈ 1:n
        @inbounds LL[i] = sum(EVI(E, D))
    end
    W = exp.(LL .- maximum(LL))
    E.W .= log.(W ./ sum(W))
    return E
end

"Learns the weights of the Ensemble by Expectation-Maximization."
function learn_ensemble_em!(E::Ensemble{T}, D::DataFrame; maxiter::Integer = 100,
        reuse::Union{Matrix{Float64}, Nothing} = nothing,
        verbose::Bool = true)::Ensemble{T} where T <: ProbCircuit
    N, K = nrow(D), length(E.C)
    ln_N = log(N)
    W = Matrix{Float64}(undef, N, K)
    N_k = Vector{Float64}(undef, K)
    ll = Vector{Float64}(undef, N)
    if isnothing(reuse)
        verbose && println("Pre-computing component log-likelihoods...")
        LL = Matrix{Float64}(undef, N, K)
        @qthreads for i ∈ 1:K @inbounds LL[:,i] .= log_likelihood_per_instance(E.C[i], D) end
    else LL = reuse end
    for j ∈ 1:maxiter
        @qthreads for i ∈ 1:K @inbounds W[:,i] .= E.W[i] .+ LL[:,i] end
        Threads.@threads for i ∈ 1:N @inbounds W[i,:] .-= logsumexp(W[i,:]) end
        @qthreads for i ∈ 1:K @inbounds N_k[i] = logsumexp(W[:,i]) end
        E.W .= N_k .- ln_N
        @qthreads for i ∈ 1:K @inbounds W[:,i] .= LL[:,i] .+ E.W[i] end
        Threads.@threads for i ∈ 1:N @inbounds ll[i] = logsumexp(W[i,:]) end
        verbose && println("EM Iteration ", j, "/", maxiter, ". Log likelihood ", sum(ll))
    end
    return E
end

"Learns the weights of the Ensemble by Stacking, with `k` as the number of folds in k-fold."
function learn_ensemble_stacking!(E::Ensemble{T}, D::DataFrame; maxiter::Integer = 100,
        k::Integer = min(nrow(D), 5), pseudocount::Real = 1.0,
        verbose::Bool = true)::Ensemble{T} where T <: ProbCircuit
    N, K = nrow(D), length(E.C)
    F = kfold(N, k)
    LL = Matrix{Float64}(undef, N, K)
    for j ∈ 1:k
        I, J = F[j]
        D_T, D_R = D[I,:], D[J,:]
        @qthreads for i ∈ 1:K estimate_parameters(E.C[i], D_R; pseudocount = pseudocount == 0 ? 1.0 : pseudocount) end
        @qthreads for i ∈ 1:K LL[I,i] .= log_likelihood_per_instance(E.C[i], D_T) end
        verbose && println("Stacking fold ", j, '/', k, '.')
    end
    learn_ensemble_em!(E, D; maxiter, verbose, reuse = LL)
    @qthreads for i ∈ 1:K estimate_parameters(E.C[i], D; pseudocount) end
    return E
end

function weighted_query(E::Ensemble{T}, D::DataFrame, f::Function; kwargs...)::Vector{Float64} where T <: ProbCircuit
    n, k = nrow(D), length(E.C)
    M = Matrix{Float64}(undef, n, k)
    @qthreads for j ∈ 1:k
        ll = f(E.C[j], D; kwargs...)
        @inbounds ll .+= E.W[j]
        for i ∈ 1:n @inbounds M[i,j] = ll[i] end
    end
    return logsumexp.(eachrow(M))
end

@inline function log_likelihood_per_instance(E::Ensemble{T}, D::DataFrame;
        use_gpu::Bool = false)::Vector{Float64} where T <: ProbCircuit
    return weighted_query(E, D, log_likelihood_per_instance; use_gpu)
end

@inline function marginal(E::Ensemble{T}, D::DataFrame)::Vector{Float64} where T <: ProbCircuit
    return weighted_query(E, D, marginal)
end
