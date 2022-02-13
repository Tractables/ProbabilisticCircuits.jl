export InputDist, Indicator, LiteralDist, BernoulliDist, CategoricalDist

abstract type InputDist end

#####################
# **Important notes for developers**
#####################

# Whenevery adding a new instance of InputDist, please specify an *UNIQUE* id using `dist_type_id`.
# The bit circuit needs this information to encode different types of input nodes.
# In order for the bit circuit code to work for new distributions, please also modify the following:
# 1. In this file, add methods: 
#    - dist_type_id
#    - num_parameters
#    - num_bpc_parameters
# 2. In queries/bit_circuit.jl, add code around lines 355 and 285
# 3. In queries/likelihood.jl, add code around line 52
# 4. In queries/flow.jl, add code around line line 177
# 5. In queries/em.jl, modify the following functions:
#    - add_pseudocount_input_kernel
#    - aggr_node_flows_input_kernel
#    - update_params_input_kernel
# Cheers!

#####################
# indicators or logical literals
#####################

"A logical literal input distribution node"
struct Indicator{T} <: InputDist
    value::T
end

const LiteralDist = Indicator{Bool}

num_parameters(n::Indicator, independent) = 1 # set to 1 since we need to store the value
num_bpc_parameters(n::Indicator) = 1 

value(d) = d.value

bits(d::Indicator, _ = nothing) = d

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

num_bpc_parameters(n::BernoulliDist) = 2

logp(d::BernoulliDist) = d.logp

bits(d::BernoulliDist, _ = nothing) = d

#####################
# categorical with more than two values
#####################

struct PolytomousDist <: CategoricalDist
    logps::Vector{Float32}
end

PolytomousDist(num_cats::Int) =
    PolytomousDist(loguniform(num_cats)gps)

num_bpc_parameters(n::PolytomousDist) = num_categories(n.logps)

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