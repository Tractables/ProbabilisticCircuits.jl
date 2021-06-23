# PROBABILISTIC CIRCUITS LIBRARY ROOT

module ProbabilisticCircuits

# USE EXTERNAL MODULES

using Reexport

using LogicCircuits
# only reexport selectively from LogicCircuits
export pos_literals, neg_literals
# circuit queries
export issmooth, isdecomposable, isstruct_decomposable, 
       isdeterministic, iscanonical
# circuit status
export num_edges, num_parameters
# datasets
export twenty_datasets

include("Utils/Utils.jl")
@reexport using .Utils

include("FactorGraph/factor_graph.jl")
include("FactorGraph/fg_compile.jl") 

include("abstract_prob_nodes.jl")
include("shared_prob_nodes.jl")
include("plain_prob_nodes.jl")
include("structured_prob_nodes.jl")
include("logistic_nodes.jl")
include("param_bit_circuit.jl")
include("param_bit_circuit_pair.jl")
include("parameters.jl")
include("gradient_based_learning.jl")

include("queries/likelihood.jl")
include("queries/marginal_flow.jl")
include("queries/map.jl")
include("queries/sample.jl")
include("queries/pr_constraint.jl")
include("queries/information.jl")
include("queries/expectation_rec.jl")
include("queries/expectation_graph.jl")
include("queries/expectation_bit.jl")

include("Logistic/Logistic.jl")
@reexport using .Logistic

include("mixtures/em.jl")

include("structurelearner/chow_liu_tree.jl")
include("structurelearner/init.jl")
include("structurelearner/heuristics.jl")
include("structurelearner/learner.jl")
include("structurelearner/vtree_learner.jl")
include("structurelearner/sample_psdd.jl")

include("ensembles/ensembles.jl")
include("ensembles/bmc.jl")

include("LoadSave/LoadSave.jl")
@reexport using .LoadSave

using Requires

function __init__()
    # optional dependency
    @require BlossomV = "6c721016-9dae-5d90-abf6-67daaccb2332" include("structurelearner/vtree_learner_blossomv.jl")
end

end
