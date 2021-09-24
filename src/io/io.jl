using LogicCircuits
using LogicCircuits: JuiceTransformer, dimacs_comments, zoo_version

using Pkg.Artifacts
using Lerche: Lerche, Lark, Transformer, @rule, @inline_rule

include("psdd_io.jl")
include("plot.jl")

# TODO CLEANUP
# include("circuit_loaders.jl")
# include("circuit_savers.jl")


#  when asked to parse/read as `ProbCircuit`, default to `PlainProbCircuit`

Base.parse(::Type{ProbCircuit}, args...) = 
    parse(PlainProbCircuit, args...)

Base.read(io::IO, ::Type{ProbCircuit},  args...) = 
    read(io, PlainProbCircuit,  args...)
