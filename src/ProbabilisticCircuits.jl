# PROBABILISTIC CIRCUITS LIBRARY ROOT

module ProbabilisticCircuits

# USE EXTERNAL MODULES

using Reexport
using LogicCircuits

include("Utils/Utils.jl")
@reexport using .Utils

include("prob_nodes.jl")
include("structured_prob_nodes.jl")
include("exp_flows.jl")
include("queries.jl")
include("informations.jl")
include("parameters.jl")

include("logistic/logistic_nodes.jl")
include("logistic/queries.jl")

include("reasoning/expectation.jl")
include("reasoning/exp_flow_circuits.jl")

include("mixtures/shared_prob_nodes.jl")
include("mixtures/em.jl")

include("structurelearner/chow_liu_tree.jl")
include("structurelearner/init.jl")
include("structurelearner/heuristics.jl")
include("structurelearner/learner.jl")


include("LoadSave/LoadSave.jl")
@reexport using .LoadSave

end
