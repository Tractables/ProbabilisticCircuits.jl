module Reasoning

using ..Logical
using ..Probabilistic
using ..Logistic
using ..Data
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