export InputDist, LiteralDist, BernoulliDist

abstract type InputDist end

#####################
# logical literals
#####################

"A logical literal input distribution node"
struct LiteralDist <: InputDist
    sign::Bool
end

num_parameters(n::LiteralDist, independent) = 0

#####################
# coin flips
#####################

"A Bernoulli input distribution node"
struct BernoulliDist <: InputDist
    logp::Float32
end

num_parameters(n::BernoulliDist, independent) = 1
