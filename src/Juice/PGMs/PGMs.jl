module PGMs

using ...Data
using ..Logical
using ..Probabilistic

export

# ChowLiuTree
learn_chow_liu_tree, parent_vector, print_tree, CLT,

# CircuitBuilder
compile_prob_circuit_from_clt, learn_prob_circuit,

# Learner
learn_vtree_from_clt, compile_psdd_from_clt,
train_mixture, initial_mixture_model_with_cluster

include("ChowLiuTree.jl")
include("CircuitBuilder.jl")
include("Learner.jl")

end