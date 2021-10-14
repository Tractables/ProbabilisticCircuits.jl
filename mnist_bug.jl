using Revise
using LogicCircuits
using DataFrames
using ProbabilisticCircuits
using StatsBase: sample
using Random
using Serialization

Random.seed!(2880)
pc = read("./mnist_b_301.jpc", ProbCircuit)
nvars = num_variables(pc)
quer = open(deserialize, "./mnist_quer.jls")
mmap_solve(pc, quer, heur="UB")
