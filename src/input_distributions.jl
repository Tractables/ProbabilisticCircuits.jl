export InputDist, LiteralDist, BernoulliDist, CategoricalDist, input_node, input_nodes

abstract type InputDist end

#####################
# **Important notes for developers**
#####################

# Whenevery adding a new instance of InputDist, please specify an *UNIQUE* id using `dist_type_id`.
# The bit circuit needs this information to encode different types of input nodes.
# In order for the bit circuit code to work for new distributions, please also modify the following:
# TBD..

#####################
# logical literals
#####################

"A logical literal input distribution node"
struct LiteralDist <: InputDist
    sign::Bool
end

dist_type_id(::LiteralDist)::UInt8 = UInt8(1)

num_parameters(n::LiteralDist, independent) = 1 # set to 1 since we need to store the sign

function input_node(::Type{<:ProbCircuit}, ::Type{LiteralDist}, var; sign::Bool = true)
    PlainInputNode(var, LiteralDist(sign))
end
function input_nodes(::Type{<:ProbCircuit}, ::Type{LiteralDist}, num_vars; sign::Bool = true)
    map(one(Var):Var(num_vars)) do v
        PlainInputNode(v, LiteralDist(sign))
    end
end

#####################
# coin flips
#####################

"A Bernoulli input distribution node"
mutable struct BernoulliDist <: InputDist
    logp::Float32
end

dist_type_id(::BernoulliDist)::UInt8 = UInt8(2)

num_parameters(n::BernoulliDist, independent) = 1

function input_node(::Type{<:ProbCircuit}, ::Type{BernoulliDist}, var; p::Float32 = 0.5)
    PlainInputNode(var, BernoulliDist(log(p)))
end
function input_nodes(::Type{<:ProbCircuit}, ::Type{BernoulliDist}, num_vars; p::Float32 = 0.5)
    map(one(Var):Var(num_vars)) do v
        PlainInputNode(v, BernoulliDist(log(p)))
    end
end

#####################
# categorical
#####################

"A Categorical input distribution node"
mutable struct CategoricalDist <: InputDist
    logps::Vector{Float32}
    CategoricalDist(num_cats) = begin
        logps = ones(Float32, num_cats) .* convert(Float32, log(1.0 / num_cats))
        new(logps)
    end
end

dist_type_id(::CategoricalDist)::UInt8 = UInt8(3)

num_parameters(n::CategoricalDist, independent) = length(n.logps)

function input_node(::Type{<:ProbCircuit}, ::Type{CategoricalDist}, var; num_cats)
    PlainInputNode(var, CategoricalDist(num_cats))
end
function input_nodes(::Type{<:ProbCircuit}, ::Type{CategoricalDist}, num_vars; num_cats)
    map(one(Var):Var(num_vars)) do v
        PlainInputNode(v, CategoricalDist(num_cats))
    end
end