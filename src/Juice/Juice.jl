#=
Circuits:
- Julia version: 1.1.1
- Author: guy
- Date: 2019-06-23
=#

module Juice

export FlowCircuit, ProbCircuit,
    AggregateFlowCircuit, XData, pass_up, log_likelihood_per_instance,
    ProbCircuit△, fully_factorized_circuit, is_decomposable,
    log_likelihood, compute_log_likelihood, node_stats, pass_down,
    FlowCircuitNode, ProbCircuitNode, LogicalCircuitNode, PosLeafNode, NegLeafNode, ⋀Node, ⋁Node,
    ProbCircuitNode, CircuitNode, FlowCircuitNode, flatmap, num_parameters,
    estimate_parameters, train_mixture, mnist, FlowCache, reset_aggregate_flows,
    Circuit△, LogicalCircuit△, FlowCircuit△, accumulate_aggr_flows_batch,
    learn_chow_liu_tree, twenty_datasets, dataset, train, WXData, feature_matrix,
    compile_prob_circuit_from_clt, num_features, clt_log_likelihood_per_instance, clt_get_log_likelihood,
    load_psdd_prob_circuit, load_circuit, circuit_matchers, LCDecisionLine,print_tree,
    parse_one_obj, LCElementTuple, BiasLine, NegLiteralLine, PosLiteralLine,
    HeaderLine, CommentLine, parse_lc_file, CircuitFormatLine,
    parse_vtree_file, compile_vtree_format_lines, load_vtree,
    VtreeNode, VtreeLeafNode, VtreeInnerNode, IsLeaf, Variables, VariableCount,
    save,
    marginal_log_likelihood_per_instance,
    marginal_pass_up_down,
    train_mixture_tree, mix_prob_circuit_check

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

# Todo(pashak) Make these IO submodule
include("IO/VtreeParser.jl")
include("IO/VtreeSaver.jl")
include("IO/CircuitParser.jl")
include("IO/CircuitSaver.jl")

# Util submodule, may need move to ../Utils/
include("Distribution.jl")

end
