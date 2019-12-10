# Driver script for all unit tests

using Jive

runtests(@__DIR__, skip=["runtests.jl", "helper"])
