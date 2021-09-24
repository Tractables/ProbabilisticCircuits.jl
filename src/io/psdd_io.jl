export zoo_psdd, zoo_psdd_file, 
    PsddFormat, PsddVtreeFormat

struct PsddFormat <: FileFormat end

const PsddVtreeFormat = Tuple{PsddFormat,VtreeFormat}
Tuple{PsddFormat,VtreeFormat}() = (PsddFormat(),VtreeFormat())

##############################################
# Read SDDs
##############################################

zoo_psdd_file(name) = 
    artifact"circuit_model_zoo" * zoo_version * "/psdds/$name"

"""
    zoo_psdd(name)

Loads PSDD file with given name from model zoo. See https://github.com/UCLA-StarAI/Circuit-Model-Zoo.    
"""
zoo_psdd(name) = 
    read(zoo_psdd_file(name), ProbCircuit, PsddFormat())

const psdd_grammar = raw"""
    start: _header (_NL node)+ _NL?

    _header : "psdd" (_WS INT)?
    
    node : "T" _WS INT _WS INT _WS INT _WS LOGPROB -> true_node
         | "L" _WS INT _WS INT _WS SIGNED_INT _WS? -> literal_node
         | "D" _WS INT _WS INT _WS INT _WS elems  -> decision_node
         
    elems : elem (_WS elem)*
    elem : INT _WS INT _WS LOGPROB

    %import common.INT
    %import common.SIGNED_INT
    %import common.SIGNED_NUMBER -> LOGPROB
    %import common.WS_INLINE -> _WS
    %import common.NEWLINE -> _NL
    """ * dimacs_comments

const psdd_parser = Lark(psdd_grammar)

abstract type PsddParse <: JuiceTransformer end

@rule start(t::PsddParse, x) = begin
    x[end]
end 

@rule elem(t::PsddParse, x) = 
    [t.nodes[x[1]], t.nodes[x[2]], Base.parse(Float64,x[3])]

@rule elems(t::PsddParse, x) = 
    Array(x)

#  parse unstructured
struct PlainPsddParse <: PsddParse
    nodes::Dict{String,PlainProbCircuit}
    PlainPsddParse() = new(Dict{String,PlainProbCircuit}())
end

@rule literal_node(t::PlainPsddParse, x) = 
    t.nodes[x[1]] = PlainProbLiteralNode(Base.parse(Lit,x[3]))

@rule true_node(t::PlainPsddParse, x) = begin
    litn = PlainProbLiteralNode(Base.parse(Lit, x[3]))
    log_prob = Base.parse(Float64, x[4])
    log_probs = [log_prob, log1p(-exp(log_prob))]
    t.nodes[x[1]] = PlainSumNode([litn, -litn], log_probs)
end

@rule decision_node(t::PlainPsddParse,x) = begin
    @assert length(x[4]) == Base.parse(Int,x[3])
    children = map(x[4]) do elem
        PlainMulNode(elem[1:2])
    end
    log_probs = map(e -> e[3], x[4])
    t.nodes[x[1]] = PlainSumNode(children, log_probs)
end

function Base.parse(::Type{PlainProbCircuit}, str, ::PsddFormat) 
    ast = Lerche.parse(psdd_parser, str)
    Lerche.transform(PlainPsddParse(), ast)
end

Base.read(io::IO, ::Type{PlainProbCircuit}, ::PsddFormat) =
    parse(PlainProbCircuit, read(io, String), PsddFormat())

#  parse structured
struct StructPsddParse <: PsddParse
    id2vtree::Dict{String,<:Vtree}
    nodes::Dict{String,StructProbCircuit}
    StructPsddParse(id2vtree) = 
        new(id2vtree,Dict{String,StructProbCircuit}())
end

@rule literal_node(t::StructPsddParse, x) = begin
    lit = Base.parse(Lit,x[3])
    vtree = t.id2vtree[x[2]]
    t.nodes[x[1]] = StructProbLiteralNode(lit, vtree)
end

@rule true_node(t::StructPsddParse, x) = begin
    vtree = t.id2vtree[x[2]]
    litn = StructProbLiteralNode(Base.parse(Lit, x[3]), vtree)
    log_prob = Base.parse(Float64, x[4])
    log_probs = [log_prob, log1p(-exp(log_prob))]
    t.nodes[x[1]] = StructSumNode([litn, -litn], log_probs, vtree)
end

@rule decision_node(t::StructPsddParse,x) = begin
    @assert length(x[4]) == Base.parse(Int,x[3])
    vtree = t.id2vtree[x[2]]
    children = map(x[4]) do elem
        StructMulNode(elem[1], elem[2], vtree)
    end
    log_probs = map(e -> e[3], x[4])
    t.nodes[x[1]] = StructSumNode(children, log_probs, vtree)
end

function Base.parse(::Type{StructProbCircuit}, str::AbstractString, ::PsddFormat, id2vtree) 
    ast = Lerche.parse(psdd_parser, str)
    Lerche.transform(StructPsddParse(id2vtree), ast)
end

function Base.parse(::Type{StructProbCircuit}, strings, format::PsddVtreeFormat) 
    id2vtree = parse(Dict{String,Vtree}, strings[2], format[2])
    parse(StructProbCircuit, strings[1], format[1], id2vtree)
end

Base.read(io::IO, ::Type{StructProbCircuit}, ::PsddFormat, id2vtree) =
    parse(StructProbCircuit, read(io, String), PsddFormat(), id2vtree)

function Base.read(ios::Tuple{IO,IO}, ::Type{StructProbCircuit}, ::PsddVtreeFormat) 
    circuit_str = read(ios[1], String)
    vtree_str = read(ios[2], String)
    parse(StructProbCircuit, (circuit_str,vtree_str), PsddVtreeFormat())
end

# ##############################################
# # Write SDDs
# ##############################################

# const SDD_FORMAT = """c this file was saved by LogicCircuits.jl
# c ids of psdd nodes start at 0
# c psdd nodes appear bottom-up, children before parents
# c
# c file syntax:
# c psdd count-of-psdd-nodes
# c F id-of-false-psdd-node
# c T id-of-true-psdd-node
# c L id-of-literal-psdd-node id-of-vtree literal
# c D id-of-decomposition-psdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub}*
# c"""

# function Base.write(io::IO, psdd::Psdd, ::PsddFormat, vtree2id::Function = (x -> 0))

#     id = -1

#     println(io, SDD_FORMAT)
#     println(io, "psdd $(psdd_num_nodes_leafs(psdd))")

#     f_con(n) = begin
#         nid = id += 1
#         sign = isfalse(n) ? "F" : "T"
#         println(io, "$sign $nid")
#         nid
#     end

#     f_lit(n) = begin
#         nid = id += 1
#         println(io, "L $nid $(vtree2id(mgr(n))) $(literal(n))")
#         nid
#     end

#     f_a(n, ids) = tuple(ids...)

#     f_o(n, ids) = begin
#         nid = id += 1
#         print(io, "D $nid $(vtree2id(mgr(n))) $(length(ids))")
#         for el in ids
#             print(io, " $(el[1]) $(el[2])")
#         end
#         println(io)
#         nid
#     end
    
#     foldup_aggregate(psdd, f_con, f_lit, f_a, f_o, Union{Int, Tuple{Int,Int}})

#     nothing
# end

# function Base.write(ios::Tuple{IO,IO}, psdd::Psdd, format::PsddVtreeFormat)
#     vtree2id = write(ios[2], mgr(psdd), format[2])
#     write(ios[1], psdd, format[1], i -> vtree2id[i])
# end