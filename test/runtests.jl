# Driver script for all unit tests

using Jive

# TODO reinstate after refactoring all modules
runtests(@__DIR__, skip=["runtests.jl", "helper", "broken", "_manual_"])
