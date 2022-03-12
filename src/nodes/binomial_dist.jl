# using SpecialFunctions: lgamma
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

init_params(dist::Binomial, perturbation::Float32) = begin
    Binomial(dist.N, rand(Float32))
end

function bits(dist::Binomial, heap)
    heap_start = length(heap) + 1
    # use heap to store parameters and space for parameter learning
    # Add (p, flow*value, flow, missing_flow, p[x=i] for i=0...N)
    append!(heap, dist.p, zeros(Float32, 3), zeros(Float32, dist.N + 1))
    BitsBinomial(dist.N, heap_start)
end

function unbits(dist::BitsBinomial, heap)
    p = heap[dist.heap_start]
    Binomial(dist.N, p)
end

function loglikelihood(dist::BitsBinomial, value, heap)
    return heap[dist.heap_start + UInt32(4) + value]
    
    # N = dist.N
    # p = heap[dist.heap_start]

    # ans = zero(Float32)
    # log_n_fact = zero(Float32)
    # for i = 1: N
    #     log_n_fact += log(i)
    #     if  i == N
    #         ans += log_n_fact # + log(n!)
    #     end
    #     if i == value
    #         ans -= log_n_fact # - log(k!)
    #     end
    #     if i==(N-value)
    #         ans -= log_n_fact # - log((n-k)!)
    #     end
    # end
    # # log_nfact(N) - log_nfact(value) - log_n_fact(N-value)

    # return ans + log(p) * value +  log1p(-p) * (N - value)
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

function log_nfact(n)
    ans = zero(Float32)
    for i = 1 : n
        ans += log(i)
    end
    return ans
end

function update_params(dist::BitsBinomial, heap, pseudocount, inertia)
    heap_start = dist.heap_start

    missing_flow = heap[heap_start + 3]
    node_flow = heap[heap_start + 2] + missing_flow + pseudocount

    oldp = heap[heap_start]
    new = (heap[heap_start + 1] + missing_flow * oldp + pseudocount) / (node_flow * dist.N)

    heap[heap_start] = oldp * inertia + new * (one(Float32) - inertia)

    # update heap for log(p[x=i]) for i=0:N
    N = dist.N
    new_p = heap[heap_start]
    
    cache_start = heap_start + UInt32(4)
    for i = UInt32(1) : N - UInt32(1)
        heap_idx = cache_start + i
        temp = log_nfact(N) - log_nfact(i) - log_nfact(N - i) 
        temp += i * log(new_p) 
        temp += (N - i) * log1p(-new_p) 
        heap[heap_idx] = temp
    end
    heap[cache_start + N] = N * log(new_p)
    heap[cache_start] = N * log1p(-new_p)
    nothing
end

function clear_memory(dist::BitsBinomial, heap, rate)
    heap_start = dist.heap_start
    for i=1 : 3
        heap[heap_start + i] *= rate
    end
    nothing
end

function sample_state(dist::BitsBinomial, threshold, heap) 
    ans::UInt32 = dist.N
    cumul_prob = typemin(Float32)
    heap_start = dist.heap_start
    N = dist.N

    cache_start = heap_start + UInt32(4)
    for i = cache_start : cache_start + N - UInt32(1)
        cumul_prob = logsumexp(cumul_prob, heap[i])
        if cumul_prob > threshold
            ans = i - cache_start
            break
        end
    end
    return ans
end