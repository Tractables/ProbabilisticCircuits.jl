# PROBABILISTIC CIRCUITS LIBRARY ROOT

module ProbabilisticCircuits

# USE EXTERNAL MODULES

using Reexport

using LogicCircuits
# only reexport selectively from LogicCircuits
export pos_literals, neg_literals

include("Utils/Utils.jl")
@reexport using .Utils

include("abstract_prob_nodes.jl")
include("plain_prob_nodes.jl")
include("structured_prob_nodes.jl")
include("logistic_nodes.jl")
include("param_bit_circuit.jl")
include("parameters.jl")

include("queries/likelihood.jl")
include("queries/marginal_flow.jl")
include("queries/map.jl")
include("queries/sample.jl")
include("queries/pr_constraint.jl")
include("queries/information.jl")
include("queries/expectation_rec.jl")
include("queries/expectation_graph.jl")

include("Logistic/Logistic.jl")
@reexport using .Logistic

include("mixtures/shared_prob_nodes.jl")
include("mixtures/em.jl")

include("structurelearner/chow_liu_tree.jl")
include("structurelearner/init.jl")
include("structurelearner/heuristics.jl")
include("structurelearner/learner.jl")
include("structurelearner/vtree_learner.jl")

include("LoadSave/LoadSave.jl")
@reexport using .LoadSave

end
