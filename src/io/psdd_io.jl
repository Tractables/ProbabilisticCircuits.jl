struct PsddFormat <: FileFormat end

##############################################
# Read SDDs
##############################################

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

psdd_parser() = Lark(psdd_grammar)

abstract type PsddParse <: PCTransformer end

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

@rule literal_node(t::PlainPsddParse, x) = begin
    lit = Base.parse(Int,x[3])
    var = abs(lit)
    sign = lit > 0
    t.nodes[x[1]] = PlainInputNode(var, LiteralDist(sign))
end

@rule true_node(t::PlainPsddParse, x) = begin
    var = Base.parse(Int,x[3])
    pos = PlainInputNode(var, LiteralDist(true))
    neg = PlainInputNode(var, LiteralDist(false))
    log_prob = Base.parse(Float64, x[4])
    log_probs = [log_prob, log1p(-exp(log_prob))]
    t.nodes[x[1]] = PlainSumNode([pos, neg], log_probs)
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
    ast = Lerche.parse(psdd_parser(), str)
    Lerche.transform(PlainPsddParse(), ast)
end

Base.read(io::IO, ::Type{PlainProbCircuit}, ::PsddFormat) =
    parse(PlainProbCircuit, read(io, String), PsddFormat())

##############################################
# Write PSDDs
##############################################

const PSDD_FORMAT = """c this file was saved by ProbabilisticCircuits.jl
c ids of psdd nodes start at 0
c psdd nodes appear bottom-up, children before parents
c
c file syntax:
c psdd count-of-sdd-nodes
c L id-of-literal-sdd-node id-of-vtree literal
c T id-of-trueNode-sdd-node id-of-vtree variable log(litProb)
c D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub log(elementProb)}*
c"""

"Count the number of decision and leaf nodes in the PSDD"
psdd_num_nodes_leafs(psdd) = length(sumnodes(psdd)) + length(inputnodes(psdd)) # defined in sdd file format

function Base.write(io::IO, pc::ProbCircuit, ::PsddFormat, vtree2id::Function = (x -> 0))

    id = -1

    println(io, PSDD_FORMAT)
    println(io, "psdd $(psdd_num_nodes_leafs(pc))")

    f_lit(n) = begin
        nid = id += 1
        literal = value(dist(n)) ? randvar(n) : -randvar(n)
        println(io, "L $nid $(vtree2id(n)) $literal")
        nid
    end

    f_a(n, ids) = begin
        if length(ids) != 2 
            error("The PSDD file format requires multiplications/AND nodes to have exactly two inputs")
        end
        tuple(ids...)
    end

    f_o(n, ids) = begin
        nid = id += 1
        vtreeid = vtree2id(n)
        if num_children(n) == 2 && all(isinput, inputs(n))
            pos_child = value(dist(children(n)[1])) > 0 ? 1 : 2 
            log_prob = params(n)[pos_child]
            v = randvar(children(n)[1])
            print(io, "T $nid $vtreeid $v $log_prob")
        else
            print(io, "D $nid $vtreeid $(length(ids))")
            for (el, log_prob) in zip(ids, params(n))
                print(io, " $(el[1]) $(el[2]) $log_prob")
            end
        end
        println(io)
        nid
    end
    
    foldup_aggregate(pc, f_lit, f_a, f_o, Union{Int, Tuple{Int,Int}})

    nothing
end