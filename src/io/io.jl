using Lerche: Lerche, Lark, Transformer, @rule, @inline_rule


#  by default don't transform tokens in parser
abstract type PCTransformer <: Transformer end

Lerche.visit_tokens(t::PCTransformer) = false

# file formats supported by this package
abstract type FileFormat end

struct GzipFormat <: FileFormat 
    inner_format::FileFormat
end

# usual comment format for DIMACS-based file formats
const dimacs_comments = raw"""
    COMMENT : ("c" | "cc") (_WS /[^\n]/*)? (_NL | /$/)
    %ignore COMMENT
"""

# if no circuit file format is given on read, infer file format from extension

function file2pcformat(file) 
    if endswith(file,".gz")
        file_inner, _ = splitext(file)
        format_inner = file2pcformat(file_inner)
        GzipFormat(format_inner)
    elseif endswith(file,".jpc")
        JpcFormat()
    elseif endswith(file,".psdd")
        PsddFormat()
    elseif endswith(file,".spn")
        SpnFormat()
    else
        throw("Unknown file extension in $file: provide a file format argument")
    end
end

"""
    Base.read(file::AbstractString, ::Type{C}) where C <: ProbCircuit

Reads circuit from file; uses extension to detect format type, for example ".psdd" for PSDDs.
"""
Base.read(file::AbstractString, ::Type{C}) where C <: ProbCircuit =
    read(file, C, file2pcformat(file))

"""
    Base.write(file::AbstractString, circuit::ProbCircuit)

Writes circuit to file; uses file name extention to detect file format.
"""
Base.write(file::AbstractString, circuit::ProbCircuit) =
    write(file, circuit, file2pcformat(file))

#  when asked to parse/read as `ProbCircuit`, default to `PlainProbCircuit`

Base.parse(::Type{ProbCircuit}, args...) = 
    parse(PlainProbCircuit, args...)

Base.read(io::IO, ::Type{ProbCircuit},  args...) = 
    read(io, PlainProbCircuit,  args...)

Base.read(io::IO, ::Type{ProbCircuit}, f::GzipFormat) = 
    # avoid method ambiguity
    read(io, PlainProbCircuit,  f)

include("jpc_io.jl")
include("psdd_io.jl")
include("spn_io.jl")