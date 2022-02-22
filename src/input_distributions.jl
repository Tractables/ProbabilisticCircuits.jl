using CUDA

export InputDist, Indicator, Literal, Bernoulli, Categorical, loglikelihood

abstract type InputDist end

import Base: isapprox #extend

isapprox(x::InputDist, y::InputDist) = 
    typeof(x) == typeof(y) && params(x) ≈ params(y)

#####################
# indicators or logical literals
#####################

"A input distribution node that places all probability on a single value"
struct Indicator{T} <: InputDist
    value::T
end

"A logical literal input distribution node"
const Literal = Indicator{Bool}

num_parameters(n::Indicator, independent) = 0

value(d::Indicator) = d.value

params(d::Indicator) = value(d)

bits(d::Indicator, _ = nothing) = d

unbits(d::Indicator, _ = nothing) = d

loglikelihood(d::Indicator, value, _ = nothing) =
    (d.value == value) ?  zero(Float32) : -Inf32

init_params(d::Indicator, _) = d

map_loglikelihood(d::Indicator, _= nothing) =
    zero(Float32)

map_state(d::Indicator, _ = nothing) = 
    d.value

sample_state(d::Indicator, _ = nothing) = 
    d.value

# no learning necessary for indicator distributions
flow(d::Indicator, value, node_flow, heap) = nothing
update_params(d::Indicator, heap, pseudocount, inertia) = nothing
clear_memory(d::Indicator, heap, rate) = nothing

#####################
# categoricals or bernoullis
#####################

"A N-value categorical input distribution ranging over integers [0...N-1]"
struct Categorical <: InputDist
    logps::Vector{Float32}
end

loguniform(num_cats) = 
    zeros(Float32, num_cats) .- log(num_cats) 

Categorical(num_cats::Integer) =
    Categorical(loguniform(num_cats))

Bernoulli() = Categorical(2)
Bernoulli(logp) = 
    Categorical([log1p(-exp(logp)), logp])
    
logps(d::Categorical) = d.logps

params(d::Categorical) = logps(d)

num_categories(d::Categorical) = length(logps(d))

num_parameters(n::Categorical, independent) = 
    num_categories(n) - (independent ? 1 : 0)

init_params(d::Categorical, perturbation::Float32) = begin
    unnormalized_probs = map(rand(Float32, num_categories(d))) do x 
        Float32(1.0 - perturbation + x * 2.0 * perturbation)
    end
    logps = log.(unnormalized_probs ./ sum(unnormalized_probs))
    Categorical(logps)
end

struct BitsCategorical
    num_cats::UInt32
    heap_start::UInt32
end

function bits(d::Categorical, heap) 
    num_cats = num_categories(d)
    heap_start = length(heap) + 1
    # use heap to store parameters and space for parameter learning
    append!(heap, logps(d), zeros(eltype(heap), num_cats + 1)) # the last value is used to maintain `missing` flows
    BitsCategorical(num_cats, heap_start)
end

function unbits(d::BitsCategorical, heap) 
    logps = heap[d.heap_start : d.heap_start + d.num_cats-one(UInt32)]
    Categorical(logps)
end

loglikelihood(d::BitsCategorical, value, heap) =
    heap[d.heap_start + UInt32(value)]

function flow(d::BitsCategorical, value, node_flow, heap)
    if ismissing(value)
        CUDA.@atomic heap[d.heap_start+UInt32(2)*d.num_cats] += node_flow
    else
        CUDA.@atomic heap[d.heap_start+d.num_cats+UInt32(value)] += node_flow
    end
    nothing
end

function update_params(d::BitsCategorical, heap, pseudocount, inertia)
    heap_start = d.heap_start
    num_cats = d.num_cats
    
    @inbounds begin
        # add pseudocount & accumulate node flow
        node_flow = zero(Float32)
        cat_pseudocount = pseudocount / Float32(num_cats)
        for i = 0 : num_cats-1
            node_flow += heap[heap_start+num_cats+i]
        end
        missing_flow = heap[heap_start+UInt32(2)*num_cats]
        node_flow += missing_flow + pseudocount
        
        # update parameter
        for i = 0 : num_cats-1
            oldp = exp(heap[heap_start+i])
            old = inertia * oldp
            new = (one(Float32) - inertia) * (heap[heap_start+num_cats+i] + 
                    cat_pseudocount + missing_flow * oldp) / node_flow 
            new_log_param = log(old + new)
            heap[heap_start+i] = new_log_param
        end
    end
    nothing
end

function clear_memory(d::BitsCategorical, heap, rate)
    heap_start = d.heap_start
    num_cats = d.num_cats
    for i = 0 : num_cats-1
        heap[heap_start+num_cats+i] *= rate
    end
    heap[heap_start+2*num_cats] *= rate
    nothing
end