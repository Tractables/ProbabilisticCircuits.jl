export InputDist, LiteralDist, BernoulliDist, CategoricalDist, input_node, input_nodes

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
# logical literals
#####################

"A logical literal input distribution node"
struct LiteralDist <: InputDist
    sign::Bool
end

num_parameters(n::LiteralDist, independent) = 1 # set to 1 since we need to store the sign
num_bpc_parameters(n::LiteralDist) = 1 

sign(d::LiteralDist) = d.sign

#####################
# coin flips
#####################

"A Bernoulli input distribution node"
mutable struct BernoulliDist <: InputDist
    logp::Float32
end

BernoulliDist() = BernoulliDist(log(0.5))

num_parameters(n::BernoulliDist, independent) = 1
num_bpc_parameters(n::BernoulliDist) = 2

logp(d::BernoulliDist) = d.logp

#####################
# categorical
#####################

"A Categorical input distribution node"
mutable struct CategoricalDist <: InputDist
    logps::Vector{Float32}
end

function CategoricalDist(num_cats::Int)
    logps = zeros(Float32, num_cats) .- log(num_cats) 
    CategoricalDist(logps)
end

num_parameters(n::CategoricalDist, independent) = 
    num_categories(n) - independent ? 1 : 0
num_bpc_parameters(n::CategoricalDist) = num_categories(n.logps)

logps(d::CategoricalDist) = d.logps

num_categories(d::CategoricalDist) = length(logps(d))
