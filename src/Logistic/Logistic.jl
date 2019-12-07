module Logistic

using StatsFuns # logsumexp
using ..Data
using ..Utils
using ..Logical

export 
    LogisticΔNode, 
    LogisticLeafNode, 
    LogisticInnerNode, 
    LogisticLiteral,
    Logistic⋀,
    Logistic⋁,
    LogisticΔ,
    LogisticΔ,
    LogisticCache,
    num_parameters_perclass,
    logistic_origin,
    class_conditional_likelihood_per_instance,
    classes



include("LogisticCircuits.jl")

end