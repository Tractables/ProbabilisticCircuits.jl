# JUICE LIBRARY ROOT


module Juice

# USE EXTERNAL MODULES

using Reexport


include("Utils/Utils.jl")
include("Data/Data.jl")

@reexport using .Data
@reexport using .Utils

# INCLUDE CHILD MODULES
include("Logical/Logical.jl")
include("Probabilistic/Probabilistic.jl")
include("Logistic/Logistic.jl")
include("IO/IO.jl")
include("StructureLearner/StructureLearner.jl")
include("Reasoning/Reasoning.jl")


# USE CHILD MODULES (in order to re-export some functions)
@reexport using .Logical
@reexport using .Probabilistic
@reexport using .IO
@reexport using .Logistic
@reexport using .StructureLearner
@reexport using .Reasoning

end
