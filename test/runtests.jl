# Driver script for all unit tests

# using Distributed

# Load Juice code on all processes
# @everywhere include("./src/Juice.jl")
# this is now handled by having Juice be a package

using Jive
# runtests("", 
#          skip=["run.jl"],
#          targets=ARGS)

runtests(@__DIR__, 
         skip=["run.jl", "helper"], 
         targets=map(x -> replace(x, "" => "", count=1), 
         ARGS))
