using LogicCircuits
using LogicCircuits: JuiceTransformer, dimacs_comments, zoo_version

using Pkg.Artifacts
using Lerche: Lerche, Lark, Transformer, @rule, @inline_rule

include("jpc_io.jl")
include("psdd_io.jl")
include("spn_io.jl")
include("clt_io.jl")
include("ensemble_io.jl")
include("plot.jl")

# if no logic circuit file format is given on read, infer file format from extension

function file2pcformat(file) 
    if endswith(file,".jpc")
        JpcFormat()
    elseif endswith(file,".psdd")
        PsddFormat()
    elseif endswith(file,".spn")
        SpnFormat()
    else
        # try a logic circuit format
        LogicCircuits.file2logicformat(file)
    end
end

"""
    Base.read(file::AbstractString, ::Type{C}) where C <: ProbCircuit

Reads circuit from file; uses extension to detect format type, for example ".psdd" for PSDDs.
"""
Base.read(file::AbstractString, ::Type{C}) where C <: ProbCircuit =
    read(file, C, file2pcformat(file))


Base.read(files::Tuple{AbstractString,AbstractString}, ::Type{C}) where C <: StructProbCircuit =
    read(files, C, (file2pcformat(files[1]), VtreeFormat()))

"""
    Base.write(file::AbstractString, circuit::ProbCircuit)

Writes circuit to file; uses file name extention to detect file format.
"""
Base.write(file::AbstractString, circuit::ProbCircuit) =
    write(file, circuit, file2pcformat(file))

"""
    Base.write(files::Tuple{AbstractString,AbstractString}, circuit::StructProbCircuit)

Saves circuit and vtree to file.
"""
Base.write(files::Tuple{AbstractString,AbstractString}, 
           circuit::StructProbCircuit) =
    write(files, circuit, (file2pcformat(files[1]), VtreeFormat()))


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