#=
Circuits:
- Julia version: 1.1.1
- Author: guy
- Date: 2019-06-23
=#

module Juice

export

# Data and Utils
XData, WXData, mnist, twenty_dataset_names, twenty_datasets, dataset, train, feature_matrix,
num_features, flatmap, num_examples, XBatches,

##################### Circuits submodule #####################
# LogicalCircuits
Var, Lit, var2lit, lit2var, CircuitNode, Circuit△, LogicalCircuitNode, LogicalCircuit△, PosLeafNode,NegLeafNode,
⋁Node, ⋀Node, num_children, children, NodeType, Inner, Leaf, ode_stats, is_decomposable, fully_factorized_circuit,

# ProbCircuits
ProbCircuitNode, ProbCircuit, ProbCircuit△, ProbLeafNode, ProbInnerNode, ProbPosLeaf,
ProbNegLeaf, Prob⋀, Prob⋁, ProbCache, cvar, num_parameters, compute_log_likelihood,
log_likelihood, estimate_parameters, log_likelihood_per_instance, marginal_log_likelihood_per_instance,
initial_mixture_model, estimate_parameters_from_aggregates, compute_ensemble_log_likelihood,
expectation_step, maximization_step, expectation_step_batch, train_mixture_with_structure,

# FlowCircuits
FlowCircuitNode, FlowCircuit, FlowCircuit△, FlowCache, pass_down, pass_up, marginal_pass_up_down,

# AggregateFlowCircuits
AggregateFlowCircuit, reset_aggregate_flows, accumulate_aggr_flows_batch, opts_accumulate_flows,

# Vtree
VtreeNode, VtreeLeafNode, VtreeInnerNode, isleaf, variables, num_variables, Vtree,
order_nodes_leaves_before_parents, VtreeLearnerContext, construct_top_down, construct_bottom_up,
isequal, isequal_unordered, left_most_child,


##################### Learning submodule #####################
# ProbMixtures
train_mixture, initial_mixture_model_with_cluster,

# ChowLiuTree
learn_chow_liu_tree, parse_clt, parent_vector, clt_log_likelihood_per_instance, clt_get_log_likelihood,
print_tree,

# CircuitBuilder
compile_prob_circuit_from_clt, prob_circuit_check, mix_prob_circuit_check, learn_prob_circuit,

# VtreeLearner
to_long_mi, MetisContext, metis_top_down, BlossomContext, blossom_bottom_up!, TestContext,
test_top_down, test_bottom_up!, learn_vtree_from_clt,

# PSDDLearner
compile_psdd_from_clt, partical_copy, satisfy, imply, make_node, calculate_all_bases,
add_correspondence, VtreeCache,

##################### IO submodule #####################
# CircuitParser
CircuitFormatLine, CommentLine, HeaderLine, PosLiteralLine, NegLiteralLine, LCElementTuple,
LCDecisionLine, BiasLine, circuit_matchers, parse_one_obj, load_circuit, parse_lc_file,
load_psdd_prob_circuit,

# CircuitSaver
get_nodes_level, save_as_dot,

# VtreeParser / Saver
parse_vtree_file, compile_vtree_format_lines, load_vtree, save,

##################### Util submodule #####################
# Distribution
calculate_all_distributions, get_cpt, mutual_information, set_mi

using Query
using IterTools
using EponymTuples
using StatsFuns

include("../Utils/Utils.jl")
include("../Data/Data.jl")

using .Data
using .Utils

# Todo(pashak) Make these Circuits submodule
include("LogicalCircuits.jl")
include("FlowCircuits.jl")
include("AggregateFlowCircuits.jl")
include("ProbCircuits.jl")
include("Vtree.jl")

# Todo(pashak) Make these Learning submodule
include("ProbMixtures.jl")
include("ChowLiuTree.jl")
include("CircuitBuilder.jl")
include("VtreeLearner.jl")
include("PSDDLearner.jl")

# Todo(pashak) Make these IO submodule
include("IO/VtreeParser.jl")
include("IO/VtreeSaver.jl")
include("IO/CircuitParser.jl")
include("IO/CircuitSaver.jl")

# Util submodule, may need move to ../Utils/
include("Distribution.jl")

end
