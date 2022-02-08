export InputDist, LiteralDist, BernoulliDist, CategoricalDist, input_nodes

abstract type InputDist end

#####################
# logical literals
#####################

"A logical literal input distribution node"
struct LiteralDist <: InputDist
    sign::Bool
end

num_parameters(n::LiteralDist, independent) = 0

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

num_parameters(n::BernoulliDist, independent) = 1

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
    CategoricalDist(num_cats::Int) = begin
        logps = ones(Float32, num_cats) * log(1.0 / num_cats)
        new(logps)
    end
end

num_parameters(n::CategoricalDist, independent) = length(n.logps)

function input_nodes(::Type{<:ProbCircuit}, ::Type{CategoricalDist}, num_vars; num_cats::Int)
    map(one(Var):Var(num_vars)) do v
        PlainInputNode(v, CategoricalDist(num_cats))
    end
end