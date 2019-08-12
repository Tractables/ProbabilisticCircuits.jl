#=
Circuits:
- Julia version: 1.1.1
- Author: guy
- Date: 2019-06-23
=#

module Juice

export

# Data and Utils
XData, WXData, mnist, twenty_datasets, dataset, train, feature_matrix, num_features,
flatmap,

##################### Circuits submodule #####################
# LogicalCircuits
Var, CircuitNode, Circuit△, LogicalCircuitNode, LogicalCircuit△, PosLeafNode,NegLeafNode,
⋁Node, ⋀Node, node_stats, is_decomposable, fully_factorized_circuit,

# ProbCircuits
ProbCircuitNode, ProbCircuit, ProbCircuit△, num_parameters, compute_log_likelihood,
log_likelihood, estimate_parameters, log_likelihood_per_instance, marginal_log_likelihood_per_instance,

# FlowCircuits
FlowCircuitNode, FlowCircuit, FlowCircuit△, FlowCache, pass_down, pass_up, marginal_pass_up_down,

# AggregateFlowCircuits
AggregateFlowCircuit, reset_aggregate_flows, accumulate_aggr_flows_batch,

# Vtree
VtreeNode, VtreeLeafNode, VtreeInnerNode, isleaf, variables, num_variables, Vtree,
order_nodes_leaves_before_parents, VtreeLearnerContext, construct_top_down, construct_bottom_up,


##################### Learning submodule #####################
# ProbMixtures
train_mixture,

# ChowLiuTree / TreeMixtures
learn_chow_liu_tree, clt_log_likelihood_per_instance, clt_get_log_likelihood,
print_tree, train_mixture_tree,

# CircuitBuilder
compile_prob_circuit_from_clt, mix_prob_circuit_check,

# VtreeLearner
to_long_mi, MetisContext, metis_top_down, BlossomContext, blossom_bottom_up!, TestContext,
test_top_down, test_bottom_up!,

##################### IO submodule #####################
# CircuitParser
CircuitFormatLine, CommentLine, HeaderLine, PosLiteralLine, NegLiteralLine, LCElementTuple,
LCDecisionLine, BiasLine, circuit_matchers, parse_one_obj, load_circuit, parse_lc_file,
load_psdd_prob_circuit,

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
include("TreeMixtures.jl")
include("VtreeLearner.jl")

# Todo(pashak) Make these IO submodule
include("IO/VtreeParser.jl")
include("IO/VtreeSaver.jl")
include("IO/CircuitParser.jl")
include("IO/CircuitSaver.jl")

# Util submodule, may need move to ../Utils/
include("Distribution.jl")

end
