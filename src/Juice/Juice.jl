# JUICE LIBRARY ROOT

# INCLUDE SIBLING MODULES
include("../Utils/Utils.jl")
include("../Data/Data.jl")

module Juice

# USE EXTERNAL MODULES

using Reexport

# using IterTools
# using EponymTuples
# using StatsFuns


# USE SIBLING MODULES
@reexport using ..Data

# INCLUDE CHILD MODULES
include("Logical/Logical.jl")
include("Probabilistic/Probabilistic.jl")
include("Logistic/Logistic.jl")
include("PGMs/PGMs.jl")
include("IO/IO.jl")

# USE CHILD MODULES (in order to re-export some functions)
@reexport using .Logical
@reexport using .Probabilistic
@reexport using .IO
@reexport using .Logistic
@reexport using .PGMs

end
