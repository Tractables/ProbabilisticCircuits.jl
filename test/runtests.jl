# Driver script for all unit tests
using Distributed

# Load Juice code on all processes
@everywhere include("../src/Juice/Juice.jl")

using Jive
runtests(@__DIR__)