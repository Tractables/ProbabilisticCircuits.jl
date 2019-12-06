# Driver script for all unit tests

using Distributed
# Load Juice code on all processes
@everywhere include("./src/Juice.jl")

using Jive
# runtests("./test/", 
#          skip=["run.jl"],
#          targets=ARGS)

runtests(@__DIR__, skip=["run.jl"], targets=map(x -> replace(x, "test/" => "", count=1), ARGS))
