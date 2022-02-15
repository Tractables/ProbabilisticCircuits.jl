using CUDA

export InputDist, Indicator, LiteralDist, BernoulliDist, CategoricalDist, loglikelihood

abstract type InputDist end

#####################
# indicators or logical literals
#####################

"A logical literal input distribution node"
struct Indicator{T} <: InputDist
    value::T
end

const LiteralDist = Indicator{Bool}

num_parameters(n::Indicator, independent) = 1 # set to 1 since we need to store the value

value(d) = d.value

bits(d::Indicator, _ = nothing) = d

unbits(d::Indicator, _ = nothing) = d

loglikelihood(d::Indicator, value, _ = nothing) =
    (d.value == value) ?  zero(Float32) : -Inf32

flow(d::Indicator, value, node_flow, heap) = nothing

update_params(d::Indicator, heap, pseudocount, inertia) = nothing

clear_memory(d::Indicator, heap, rate) = nothing

#####################
# categorical
#####################

"A Categorical input distribution node"
abstract type CategoricalDist <: InputDist end

function CategoricalDist(logps::Vector)
    @assert sum(exp.(logps)) ≈ 1
    if length(logps) == 2
        BernoulliDist(logps[2])
    else
        @assert length(logps) > 2 "Categorical distributions need at least 2 values"
        PolytomousDist(logps)
    end
end

loguniform(num_cats) = 
    zeros(Float32, num_cats) .- log(num_cats) 

CategoricalDist(num_cats::Integer) =
    CategoricalDist(loguniform(num_cats))

num_parameters(n::CategoricalDist, independent) = 
    num_categories(n) - independent ? 1 : 0

#####################
# coin flips
#####################

"A Bernoulli input distribution node"
struct BernoulliDist <: CategoricalDist
    # 1/ note that we special case Bernoullis from Categoricals in order to 
    # perhaps speed up memory loads on the GPU, since the logp here does not need a pointer
    # 2/ note that containers of BernoulliDist are mutable, so this struct can remain immutable and isbits
    logp::Float32
end

BernoulliDist() = BernoulliDist(log(0.5))

num_categories(::BernoulliDist) = 2

logp(d::BernoulliDist) = d.logp


loglikelihood(d::BernoulliDist, value) =
    isone(value) ? d.logp : log1p(-exp(d.logp))

struct BitsBernoulliDist
    heap_start::UInt32
end

function bits(d::BernoulliDist, heap)
    logp = d.logp
    heap_start = length(heap) + 1
    # use heap to store parameters and space for parameter learning
    push!(heap, logp)
    append!(heap, zeros(eltype(heap), 2))
    BitsBernoulliDist(heap_start)
end

function unbits(d::BitsBernoulliDist, heap)
    logp = heap[d.heap_start]
    BernoulliDist(logp)
end

loglikelihood(d::BitsBernoulliDist, value, heap) =
    isone(value) ? heap[d.heap_start] : log1p(-exp(heap[d.heap_start]))

flow(d::BitsBernoulliDist, value, node_flow, heap) = begin
    if isone(value)
        CUDA.@atomic heap[d.heap_start+2] += node_flow
    else
        CUDA.@atomic heap[d.heap_start+1] += node_flow
    end
    nothing
end

update_params(d::BitsBernoulliDist, heap, pseudocount, inertia) = begin
    heap_start = d.heap_start
    
    # add pseudocount
    heap[heap_start+1] += pseudocount * Float32(0.5)
    heap[heap_start+2] += pseudocount * Float32(0.5)
    
    # node flow
    node_flow = heap[heap_start+1] + heap[heap_start+2]
    
    # update parameter
    old = inertia * exp(heap[heap_start])
    new = (one(Float32) - inertia) * heap[heap_start+2] / node_flow 
    new_log_param = log(old + new)
    heap[heap_start] = new_log_param
    
    nothing
end

clear_memory(d::BitsBernoulliDist, heap, rate) = begin
    heap_start = d.heap_start
    heap[heap_start+1] *= rate
    heap[heap_start+2] *= rate
    nothing
end

#####################
# categorical with more than two values
#####################

struct PolytomousDist <: CategoricalDist
    logps::Vector{Float32}
end

PolytomousDist(num_cats::Int) =
    PolytomousDist(loguniform(num_cats))

logps(d::PolytomousDist) = d.logps

num_categories(d::PolytomousDist) = length(logps(d))

struct BitsPolytomousDist
    num_cats::UInt32
    heap_start::UInt32
end

function bits(d::PolytomousDist, heap) 
    num_cats = num_categories(d)
    heap_start = length(heap) + 1
    # use heap to store parameters and space for parameter learning
    append!(heap, logps(d))
    append!(heap, zeros(eltype(heap), num_cats))
    BitsPolytomousDist(num_cats, heap_start)
end

function unbits(d::BitsPolytomousDist, heap) 
    logps = heap[d.heap_start : d.heap_start+d.num_cats-1]
    PolytomousDist(logps)
end

loglikelihood(d::BitsPolytomousDist, value, heap) =
    heap[d.heap_start+value-1]

flow(d::BitsPolytomousDist, value, node_flow, heap) = begin
    CUDA.@atomic heap[d.heap_start+d.num_cats+value-1] += node_flow
    nothing
end

update_params(d::BitsPolytomousDist, heap, pseudocount, inertia) = begin
    heap_start = d.heap_start
    num_cats = d.num_cats
    
    # add pseudocount & accumulate node flow
    node_flow = zero(Float32)
    cat_pseudocount = pseudocount / Float32(num_cats)
    for i = 1 : num_cats
        heap[heap_start+num_cats-1+i] += cat_pseudocount
        node_flow += heap[heap_start+num_cats-1+i]
    end
    
    # update parameter
    for i = 1 : num_cats
        old = inertia * exp(heap[heap_start-1+i])
        new = (one(Float32) - inertia) * heap[heap_start+num_cats-1+i] / node_flow 
        new_log_param = log(old + new)
        heap[heap_start-1+i] = new_log_param
    end
    
    nothing
end

clear_memory(d::BitsPolytomousDist, heap, rate) = begin
    heap_start = d.heap_start
    num_cats = d.num_cats
    for i = 1 : num_cats
        heap[heap_start+num_cats-1+i] *= rate
    end
    nothing
end