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

# copy read/write API for tuples of files

function Base.read(files::Tuple{AbstractString, AbstractString}, ::Type{C}, args...) where C <: StructProbCircuit
    open(files[1]) do io1 
        open(files[2]) do io2 
            read((io1, io2), C, args...)
        end
    end
end

function Base.write(files::Tuple{AbstractString,AbstractString},
                    circuit::StructProbCircuit, args...) 
    open(files[1], "w") do io1
        open(files[2], "w") do io2
            write((io1, io2), circuit, args...)
        end
    end
end