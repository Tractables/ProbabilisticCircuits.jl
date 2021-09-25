export zoo_jpc, zoo_jpc_file, 
    JpcFormat, JpcVtreeFormat

struct JpcFormat <: FileFormat end

const JpcVtreeFormat = Tuple{JpcFormat,VtreeFormat}
Tuple{JpcFormat,VtreeFormat}() = (JpcFormat(),VtreeFormat())

##############################################
# Read JPC (Juice Probabilistic Circuit)
##############################################

zoo_jpc_file(name) = 
    artifact"circuit_model_zoo" * zoo_version * "/jpcs/$name"

"""
    zoo_jpc(name)

Loads JPC file with given name from model zoo. See https://github.com/UCLA-StarAI/Circuit-Model-Zoo.    
"""
zoo_jpc(name) = 
    read(zoo_jpc_file(name), ProbCircuit, JpcFormat())

const jpc_grammar = raw"""
    start: header (_NL node)+ _NL?

    header : "jpc" _WS INT
    
    node : "L" _WS INT _WS INT _WS SIGNED_INT -> literal_node
         | "S" _WS INT _WS INT _WS INT _WS child_nodes -> sum_node
         | "P" _WS INT _WS INT _WS INT _WS child_nodes _WS log_probs -> prod_node
         
    child_nodes : INT (_WS INT)*
    log_probs : LOGPROB (_WS LOGPROB)*
    
    %import common.INT
    %import common.SIGNED_INT
    %import common.SIGNED_NUMBER -> LOGPROB
    %import common.WS_INLINE -> _WS
    %import common.NEWLINE -> _NL
    """ * dimacs_comments

const jpc_parser = Lark(jpc_grammar)

abstract type JpcParse <: JuiceTransformer end

@inline_rule header(t::JpcParse, x) = 
    Base.parse(Int,x)

@rule start(t::JpcParse, x) = begin
    @assert num_nodes(x[end]) == x[1]
    x[end]
end 

@rule child_nodes(t::JpcParse, x) = 
    map(id -> t.nodes[id], x)

@rule log_probs(t::JpcParse, x) = 
    Base.parse.(Float64,x)

#  parse unstructured
struct PlainJpcParse <: JpcParse
    nodes::Dict{String,PlainProbCircuit}
    PlainJpcParse() = new(Dict{String,PlainProbCircuit}())
end

@rule literal_node(t::PlainJpcParse, x) = 
    t.nodes[x[1]] = PlainLiteralNode(Base.parse(Lit,x[3]))

@rule sum_node(t::PlainJpcParse,x) = begin
    @assert length(x[4]) == Base.parse(Int,x[3])
    t.nodes[x[1]] = PlainSumNode(x[4], x[5])
end

@rule prod_node(t::PlainJpcParse,x) = begin
    @assert length(x[4]) == Base.parse(Int,x[3])
    t.nodes[x[1]] = PlainMulNode(x[4])
end

function Base.parse(::Type{PlainProbCircuit}, str, ::JpcFormat) 
    ast = Lerche.parse(jpc_parser, str)
    Lerche.transform(PlainJpcParse(), ast)
end

Base.read(io::IO, ::Type{PlainProbCircuit}, ::JpcFormat) =
    parse(PlainProbCircuit, read(io, String), JpcFormat())

#  parse structured
struct StructJpcParse <: JpcParse
    id2vtree::Dict{String,<:Vtree}
    nodes::Dict{String,StructProbCircuit}
    StructJpcParse(id2vtree) = 
        new(id2vtree,Dict{String,StructProbCircuit}())
end

@rule literal_node(t::StructJpcParse, x) = begin
    lit = Base.parse(Lit,x[3])
    vtree = t.id2vtree[x[2]]
    t.nodes[x[1]] = StructProbLiteralNode(lit, vtree)
end


@rule sum_node(t::StructJpcParse,x) = begin
    @assert length(x[4]) == Base.parse(Int,x[3])
    vtree = t.id2vtree[x[2]]
    t.nodes[x[1]] = StructSumNode(x[4], x[5], vtree)
end

@rule prod_node(t::StructJpcParse,x) = begin
    @assert length(x[4]) == Base.parse(Int,x[3]) == 2
    vtree = t.id2vtree[x[2]]
    t.nodes[x[1]] = StructMulNode(x[4]..., vtree)
end

function Base.parse(::Type{StructProbCircuit}, str::AbstractString, ::JpcFormat, id2vtree) 
    ast = Lerche.parse(jpc_parser, str)
    Lerche.transform(StructJpcParse(id2vtree), ast)
end

function Base.parse(::Type{StructProbCircuit}, strings, format::JpcVtreeFormat) 
    id2vtree = parse(Dict{String,Vtree}, strings[2], format[2])
    parse(StructProbCircuit, strings[1], format[1], id2vtree)
end

Base.read(io::IO, ::Type{StructProbCircuit}, ::JpcFormat, id2vtree) =
    parse(StructProbCircuit, read(io, String), JpcFormat(), id2vtree)

function Base.read(ios::Tuple{IO,IO}, ::Type{StructProbCircuit}, ::JpcVtreeFormat) 
    circuit_str = read(ios[1], String)
    vtree_str = read(ios[2], String)
    parse(StructProbCircuit, (circuit_str,vtree_str), JpcVtreeFormat())
end

##############################################
# Write JPCs
##############################################

const JPC_FORMAT = """c this file was saved by ProbabilisticCircuits.jl
c ids of jpc nodes start at 0
c jpc nodes appear bottom-up, children before parents
c
c file syntax:
c jpc count-of-jpc-nodes
c L id-of-literal-jpc-node id-of-vtree literal
c S id-of-sum-jpc-node id-of-vtree number-of-children {child-id}+
c P id-of-product-jpc-node id-of-vtree number-of-children {child-id}+ {log-probability}+
c"""

function Base.write(io::IO, circuit::ProbCircuit, ::JpcFormat, vtreeid::Function = (x -> 0))

    labeling = label_nodes(circuit)
    map!(x -> x-1, values(labeling)) # vtree nodes are 0-based indexed

    println(io, JPC_FORMAT)
    println(io, "jpc $(num_nodes(circuit))")
    foreach(circuit) do n
        if isliteralgate(n)
            println(io, "L $(labeling[n]) $(vtreeid(n)) $(literal(n))")
        elseif isconstantgate(n)
            @assert false
        else
            t = is⋀gate(n) ? "P" : "S"
            print(io, "$t $(labeling[n]) $(vtreeid(n)) $(num_children(n))")
            for child in children(n)
                print(io, " $(labeling[child])")
            end
            for logp in n.log_probs
                print(io, " $logp")
            end
            println(io)
        end
    end
    nothing
end

function Base.write(ios::Tuple{IO,IO}, circuit::StructProbCircuit, format::JpcVtreeFormat)
    vtree2id = write(ios[2], vtree(circuit), format[2])
    write(ios[1], circuit, format[1], n -> vtree2id[vtree(n)])
end