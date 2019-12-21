module Reasoning

using LogicCircuits
using ..Probabilistic
using ..Logistic
using ..Utils

export 
    UpExpFlow,
    ExpFlowΔ,
    exp_pass_up,
    Expectation,
    ExpectationUpward,
    Moment

include("Expectation.jl")
include("ExpFlowCircuits.jl")


end