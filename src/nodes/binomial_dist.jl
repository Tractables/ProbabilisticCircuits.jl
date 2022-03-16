using SpecialFunctions: lgamma
using CUDA

export Binomial

struct Binomial <: InputDist
    N::UInt32
    p::Float32
end

struct BitsBinomial <: InputDist
    N::UInt32
    heap_start::UInt32
end

Binomial(N::UInt32) =
    Binomial(N, Float32(0.5))

num_parameters(dist::Binomial, independent) = 1
params(dist::Binomial, independent) = dist.p

isapprox(x::Binomial, y::Binomial) = 
    typeof(x) == typeof(y) && x.N == y.N && x.p ≈ y.p

init_params(dist::Binomial, perturbation::Float32) = begin
    Binomial(dist.N, rand(Float32))
end

function bits(dist::Binomial, heap)
    heap_start = length(heap) + 1
    # use heap to store parameters and space for parameter learning
    # Add (p, flow*value, flow, missing_flow)
    append!(heap, dist.p, zeros(Float32, 3))
    BitsBinomial(dist.N, heap_start)
end

pr(dist::Binomial, _ = nothing) = dist.p
pr(dist::BitsBinomial, heap) = heap[dist.heap_start]

function unbits(dist::BitsBinomial, heap)
    Binomial(dist.N, pr(dist, heap))
end

function loglikelihood(dist::Binomial, value, _ = nothing)
    binomial_logpdf_(dist.N, pr(dist), value)
end

function loglikelihood(dist::BitsBinomial, value, heap)
    binomial_logpdf_(dist.N, pr(dist, heap), value)
end

function log_nfact(n)
    return lgamma(Float32(n + 1))
end
function binomial_logpdf_(n, p, k)
    if k > n || k < 0
        return -Inf32
    elseif (p == zero(Float32))
        return (k == 0 ? Float32(0.0) : -Inf32)

    elseif (p == one(Float32))
         return (k == n ? Float32(0.0) : -Inf32)
    else
        temp = log_nfact(n) - log_nfact(k) - log_nfact(n - k) 
        temp += k * log(p) + (n - k) * log1p(-p) 
        return Float32(temp)
    end
end

function flow(dist::BitsBinomial, value, node_flow, heap)
    heap_start = dist.heap_start
    if ismissing(value)
        CUDA.@atomic heap[heap_start + UInt32(3)] += node_flow
    else
        CUDA.@atomic heap[heap_start + UInt32(1)] += node_flow * value
        CUDA.@atomic heap[heap_start + UInt32(2)] += node_flow
    end
    nothing
end

function update_params(dist::BitsBinomial, heap, pseudocount, inertia)
    heap_start = dist.heap_start

    missing_flow = heap[heap_start + 3]
    node_flow = heap[heap_start + 2] + missing_flow + pseudocount

    oldp = heap[heap_start]
    new = (heap[heap_start + 1] + missing_flow * oldp + pseudocount) / (node_flow * dist.N)

    new_p = oldp * inertia + new * (one(Float32) - inertia)
    
    # update p on heap
    heap[heap_start] = new_p
    
    nothing
end

function clear_memory(dist::BitsBinomial, heap, rate)
    heap_start = dist.heap_start
    for i = 1 : 3
        heap[heap_start + i] *= rate
    end
    nothing
end


#### Sample
function sample_state(dist::Union{BitsBinomial, Binomial}, threshold, heap) 
    # Works for both cpu and gpu
    N = dist.N
    ans::UInt32 = N
    cumul_prob = typemin(Float32)
    for i = 0 : N
        cumul_prob = logsumexp(cumul_prob, loglikelihood(dist, i, heap))
        if cumul_prob > threshold
            ans = i
            break
        end
    end
    return ans
end

sample_state(dist::Binomial, threshold) =
    sample_state(dist, threshold, nothing)

### MAP
init_heap_map_state!(dist::BitsBinomial, heap) = nothing
init_heap_map_loglikelihood!(dist::BitsBinomial, heap) = nothing

function map_loglikelihood(dist::Union{BitsBinomial, Binomial}, heap)
    p = pr(dist, heap)
    N = dist.N
    A = floor(UInt32, N*p)
    lA = loglikelihood(dist, A, heap)

    B  = floor(UInt32, N*p + 1)
    lB = loglikelihood(dist, B, heap)
    
    return max(lA, lB)
end
map_loglikelihood(dist::Binomial) = map_loglikelihood(dist, nothing)

function map_state(dist::Union{BitsBinomial, Binomial}, heap) 
    p = pr(dist, heap)
    N = dist.N
    
    A = floor(UInt32, N*p)
    lA = loglikelihood(dist, A, heap)
    
    B = floor(UInt32, N*p + 1)
    lB = loglikelihood(dist, B, heap)

    return (lA > lB ? A : B)
end
map_state(dist::Binomial) = map_state(dist, nothing)