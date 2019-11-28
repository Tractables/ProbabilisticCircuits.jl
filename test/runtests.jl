# Driver script for all unit tests

# Load Juice code on all processes
@everywhere include("../src/Juice/Juice.jl")

using Jive
runtests(@__DIR__)