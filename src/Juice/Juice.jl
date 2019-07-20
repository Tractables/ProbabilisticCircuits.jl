#=
Circuits:
- Julia version: 1.1.1
- Author: guy
- Date: 2019-06-23
=#

module Juice

export FlowCircuit, ProbCircuit,
    AggregateFlowCircuit, XData, pass_up, log_likelihood_per_instance,
    ProbCircuit△, fully_factorized_circuit,
    log_likelihood, compute_log_likelihood, node_stats, pass_down,
    FlowCircuitNode, ProbCircuitNode, LogicalCircuitNode,
    ProbCircuitNode, CircuitNode, FlowCircuitNode, flatmap, num_parameters,
    estimate_parameters, train_mixture, mnist, FlowCache, reset_aggregate_flows,
    Circuit△, LogicalCircuit△, FlowCircuit△, accumulate_aggr_flows_batch,
    learn_chow_liu_tree, twenty_datasets, dataset, train, WXData, feature_matrix,
    compile_prob_circuit_from_clt, num_features, clt_likelihood_per_instance, get_infernece,
    load_psdd_prob_circuit, load_circuit, circuit_matchers, LCDecisionLine,
    parse_one_obj, LCElementTuple, BiasLine, NegLiteralLine, PosLiteralLine,
    HeaderLine, CommentLine, parse_lc_file, CircuitFormatLine,
    parse_vtree_file, compile_vtree_format_lines,
    VtreeNode, VtreeLeafNode, VtreeInnerNode, IsLeaf, Variables, VariableCount,
    save,
    marginal_log_likelihood_per_instance,
    marginal_pass_up_down

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

# Todo(pashak) Make these IO submodule
include("IO/VtreeParser.jl")
include("IO/CircuitParser.jl")

end
