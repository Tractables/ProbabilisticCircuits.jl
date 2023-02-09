using CUDA
import Random: default_rng

export Gaussian

struct Gaussian <: InputDist
    mu::Float32
    sigma::Float32
end

struct BitsGaussian <: InputDist
    # mu::Float32
    # sigma::Float32
    sigma::Float32
    heap_start::UInt32
end

# heap_start index offsets
const GAUSS_HEAP_MU  = UInt32(1)
const GAUSS_HEAP_FLOWVALUE = UInt32(2) # flow*value
const GAUSS_HEAP_FLOW = UInt32(3) # flow
const GAUSS_HEAP_MISSINGFLOW = UInt32(4) # missing_flow

Gaussian(mu::Float64, sigma::Float64) =
    Gaussian(Float32(mu), Float32(sigma))

num_parameters(dist::Gaussian, independent) = 1

params(dist::Gaussian) = dist.mu

init_params(dist::Gaussian, perturbation) =
    Gaussian(0.0, dist.sigma)

function bits(dist::Gaussian, heap)
    heap_start = length(heap) + 1

    # Append mu, sigma, flow*value, flow, missing_flow
    append!(heap, dist.mu, zeros(Float32, 3))
    BitsGaussian(dist.sigma, heap_start)
end

mu(dist::Gaussian, _ = nothing) = dist.mu
mu(dist::BitsGaussian, heap) = heap[dist.heap_start]

sigma(dist::Gaussian, _ = nothing) = dist.sigma

function unbits(dist::Gaussian, heap)
    Gaussian(mu(dist, heap), dist.sigma)
end
    
function loglikelihood(dist::Gaussian, value, _ = nothing)
    # normlogpdf((value - mu(dist))/sigma(dist))
    log_gauss(value, mu(dist), dist.sigma)
end

function loglikelihood(dist::BitsGaussian, value, heap)
    # normlogpdf((value - mu(dist, heap))/sigma(dist, heap))
    log_gauss(value, mu(dist, heap), dist.sigma)
end

log_gauss(x, mu, sigma) = -log(sigma) - 0.5*log(2π) - 0.5*((x - mu)/sigma)^2
    
function flow(dist::BitsGaussian, value, node_flow, heap)
    heap_start = dist.heap_start

    if ismissing(value)
        CUDA.@atomic heap[heap_start + GAUSS_HEAP_MISSINGFLOW] += node_flow
    else
        CUDA.@atomic heap[heap_start + GAUSS_HEAP_FLOWVALUE] += node_flow * value
        CUDA.@atomic heap[heap_start + GAUSS_HEAP_FLOW] += node_flow
    end
    nothing
end

function update_params(dist::BitsGaussian, heap, pseudocount, inertia) 
    heap_start = dist.heap_start

    missing_flow = heap[heap_start + GAUSS_HEAP_MISSINGFLOW]
    node_flow = heap[heap_start + GAUSS_HEAP_FLOW] + missing_flow + pseudocount

    old_mu = heap[heap_start]

    new = (heap[heap_start + GAUSS_HEAP_FLOWVALUE] + (missing_flow + pseudocount) * old_mu) / (node_flow)
    new_mu = old_mu * inertia + new * (one(Float32) - inertia)
    
    # update mu on heap
    heap[heap_start] = new_mu
    nothing  
end
    
function clear_memory(dist::BitsGaussian, heap, rate)
    heap_start = dist.heap_start
    for i = 1 : 3
        heap[heap_start + i] *= rate
    end
    nothing
end

function sample_state(dist::Union{BitsGaussian, Gaussian}, threshold, heap)
    # Sample from standard normal
    z = randn()
    
    # Reparameterize
    return dist.sigma * z + dist.mu
end

### MAP
init_heap_map_state!(dist::Gaussian, heap)  = nothing

init_heap_map_loglikelihood!(dist::Gaussian, heap) = nothing

map_state(dist::Gaussian, heap) = dist.mu

map_loglikelihood(dist::Gaussian, heap) = loglikelihood(dist, dist.mu, heap)
    
