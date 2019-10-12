module Logistic

using StatsFuns # logsumexp
using ...Data
using ...Utils
using ..Logical

export 
    LogisticCircuitNode, 
    LogisticLeafNode, 
    LogisticInnerNode, 
    LogisticLiteral,
    Logistic⋀,
    Logistic⋁,
    LogisticCircuit△,
    LogisticCircuit,
    LogisticCache,
    class_conditional_likelihood_per_instance



include("LogisticCircuits.jl")

end