module Probabilistic

using Query
using StatsFuns # logsumexp

using ...Data

using ..Logical

export

# ProbCircuits
ProbCircuitNode, ProbCircuit, ProbCircuit△, ProbLeafNode, ProbInnerNode, ProbPosLeaf,
ProbNegLeaf, Prob⋀, Prob⋁, ProbCache, cvar, num_parameters, compute_log_likelihood,
log_likelihood, estimate_parameters, log_likelihood_per_instance, marginal_log_likelihood_per_instance,
initial_mixture_model, estimate_parameters_from_aggregates, compute_ensemble_log_likelihood,
expectation_step, maximization_step, expectation_step_batch, train_mixture_with_structure, check_parameter_integrity,
ll_per_instance_per_component,
sample,
# ProbMixtures
train_mixture,
bootstrap_samples_ids, learn_mixture_bagging, learn_mixture_bagging2,

# VtreeLearner
to_long_mi, MetisContext, metis_top_down, BlossomContext, blossom_bottom_up!,
test_top_down, test_bottom_up!,

# MutualInformation
mutual_information

include("ProbCircuits.jl")
include("MutualInformation.jl")
include("ProbMixtures.jl")
include("VtreeLearner.jl")
include("Bagging.jl")

end
