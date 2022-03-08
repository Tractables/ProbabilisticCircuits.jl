
using SpecialFunctions: lgamma

struct Binomial <: InputDist
    N::UInt32
    p::Float32
end

struct BitsBinomial
    N::UInt32
    heap_start::UInt32
end

num_parameters(dist::Binomial, independent) = 1
params(dist::Binomial, independent) = dist.p

init_params(dist::Binomial, perturbation::Float32) = begin
    Binomial(dist.N, rand(Float32))
end

function bits(dist::Binomial, heap)
    heap_start = length(heap) + 1
    # use heap to store parameters and space for parameter learning
    append!(heap, dist.p, zeros(eltype(heap), 2)) # use last one for missing
    BitsBinomial(dist.N, heap_start)
end

function ubits(dist::BitsBinomial, heap)
    p = heap[dist.heap_start]
    Binomial(dist.N, p)
end

function loglikelihood(dist::BitsBinomial, value, heap)
    N = dist.n
    p = heap[dist.heap_start]
    return lgamma(N + 1) - lgamma(value + 1) - lgamma(N - value + 1) + 
        log(p) * value +  log1p(-p) * (N - value)
end

function flow(dist::BitsBinomial, value, node_flow, heap)
    if ismissing(value)
        CUDA.@atomic heap[dist.heap_start + UInt32(2)] += node_flow
    else
        CUDA.@atomic heap[dist.heap_start + UInt32(1)] += node_flow
    end
    nothing
end

function update_params(dist::BitsBinomial, heap, pseudocount, inertia)
    # TODO
    nothing
end

function clear_memory(dist::BitsBinomial, heap, rate)
    heap_start = dist.heap_start
    for i=1:2
        heap[heap_start + i] *= rate
    end
    nothing
end