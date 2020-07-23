# PROBABILISTIC CIRCUITS LIBRARY ROOT

module ProbabilisticCircuits

# USE EXTERNAL MODULES

using Reexport
using LogicCircuits

include("Utils/Utils.jl")
@reexport using .Utils


# INCLUDE CHILD MODULES
include("Probabilistic/Probabilistic.jl")
# include("Logistic/Logistic.jl")
include("LoadSave/LoadSave.jl")
# include("StructureLearner/StructureLearner.jl")
# include("Reasoning/Reasoning.jl")


# USE CHILD MODULES (in order to re-export some functions)
@reexport using .Probabilistic
# @reexport using .Logistic
@reexport using .LoadSave
# @reexport using .StructureLearner
# @reexport using .Reasoning

end
