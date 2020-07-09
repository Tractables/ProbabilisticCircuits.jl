module Logistic

using LogicCircuits
using ..Utils

export 
    LogisticNode, 
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