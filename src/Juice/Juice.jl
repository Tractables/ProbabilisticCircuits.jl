#=
Circuits:
- Julia version: 1.1.1
- Author: guy
- Date: 2019-06-23
=#

module Juice

export load_psdd_prob_circuit, FlowCircuit, ProbCircuit,
    AggregateFlowCircuit, XData, pass_up, log_likelihood_per_instance,
    load_circuit, ProbCircuit△, fully_factorized_circuit,
    log_likelihood, compute_log_likelihood, node_stats, pass_down,
    FlowCircuitNode, ProbCircuitNode, circuit_matchers, LCDecisionLine,
    parse_one_obj, LCElementTuple, BiasLine, NegLiteralLine, PosLiteralLine,
    HeaderLine, CommentLine, parse_lc_file, LogicalCircuitNode, CircuitFormatLine,
    ProbCircuitNode, CircuitNode, FlowCircuitNode, flatmap, num_parameters,
    estimate_parameters, train_mixture, mnist, FlowCache, reset_aggregate_flows,
    Circuit△, LogicalCircuit△, FlowCircuit△, accumulate_aggr_flows_batch, learn_chow_liu_tree,
    twenty_datasets, dataset, train, WXData

using Query
using IterTools
using EponymTuples
using StatsFuns

include("../Utils/Utils.jl")
include("../Data/Data.jl")

using .Data
using .Utils

include("LogicalCircuits.jl")
include("CircuitParser.jl")
include("FlowCircuits.jl")
include("AggregateFlowCircuits.jl")
include("ProbCircuits.jl")
include("ProbMixtures.jl")
include("ChowLiuTree.jl")

end
