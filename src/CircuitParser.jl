#=
Circuits:
- Julia version: 1.1.1
- Author: guy
- Date: 2019-06-23
=#

using ParserCombinator
using Test
using EponymTuples

#####################
# general parser infrastructure for circuits (both LC and PSDD)
#####################

abstract type CircuitFormatLine end

struct CommentLine{T<:AbstractString} <: CircuitFormatLine
    comment::T
end

struct HeaderLine <: CircuitFormatLine end

struct PosLiteralLine <: CircuitFormatLine
    node_id::UInt32
    vtree_id::UInt32
    variable::Var
    weights::Vector{Float32}
end

struct NegLiteralLine <: CircuitFormatLine
    node_id::UInt32
    vtree_id::UInt32
    variable::Var
    weights::Vector{Float32}
end

struct LiteralLine <: CircuitFormatLine
    node_id::UInt32
    vtree_id::UInt32
    literal::Lit
end

struct TrueLeafLine <: CircuitFormatLine
    node_id::UInt32
    vtree_id::UInt32
    variable::Var
    weight::Float32
end

abstract type ElementTuple end

struct LCElementTuple <: ElementTuple
    prime_id::UInt32
    sub_id::UInt32
    weights::Vector{Float32}
end

struct PSDDElementTuple <: ElementTuple
    prime_id::UInt32
    sub_id::UInt32
    weight::Float32
end

abstract type DecisionLine <: CircuitFormatLine end

struct LCDecisionLine <: DecisionLine
    node_id::UInt32
    vtree_id::UInt32
    num_elements::UInt32
    elements:: Vector{ElementTuple}
end

struct PSDDDecisionLine <: DecisionLine
    node_id::UInt32
    vtree_id::UInt32
    num_elements::UInt32
    elements:: Vector{PSDDElementTuple}
end

struct BiasLine <: CircuitFormatLine
    node_id::UInt32
    weights::Vector{Float32}
    BiasLine(weights) = new(typemax(UInt32), weights)
end

function build_circuit_matchers()
    spc::Matcher = Drop(Space())
    pfloat16::Matcher = Parse(p"-?(\d*\.?\d+|\d+\.\d*)([eE]-?\d+)?", Float32) # missing from parser library, fixes missing exponent sign

    comment::Matcher = Seq!(E"c", Drop(Space()[0:1]), Pattern(r".*"), Eos()) > CommentLine{String}

    header_correct::Matcher = Seq!(E"Logistic Circuit", Eos())
    header_typo::Matcher = Seq!(E"Logisitic Circuit", Eos())
    header::Matcher = Alt!(header_correct, header_typo) > HeaderLine

    weights::Matcher = PlusList!(pfloat16, spc) |>  Vector{Float32}

    true_literal::Matcher = Seq!(E"T", spc, PUInt32(), spc, PUInt32(), spc, PUInt32(), spc, weights, Eos()) > PosLiteralLine
    false_literal::Matcher = Seq!(E"F", spc, PUInt32(), spc, PUInt32(), spc, PUInt32(), spc, weights, Eos()) > NegLiteralLine
    literal::Matcher = Alt!(true_literal, false_literal)

    lc_element::Matcher = Seq!(PUInt32(), spc, PUInt32(), spc, weights) > LCElementTuple
    lc_element_list::Matcher = PlusList!(Seq!(E"(", lc_element, E")"), spc) |> Vector{ElementTuple}
    lc_decision::Matcher = Seq!(E"D", spc, PUInt32(), spc, PUInt32(), spc, PUInt32(), spc, lc_element_list, Eos()) > LCDecisionLine

    bias::Matcher = Seq!(E"B", spc, weights, Eos()) > BiasLine

    lc_line::Matcher = Alt!(lc_decision, literal, comment, header, bias)

    @eponymtuple(comment, header, weights, true_literal, false_literal, literal,
                 lc_element, lc_element_list, lc_decision, lc_line)
end

const circuit_matchers = build_circuit_matchers()
const parens = r"\(([^\)]+)\)"

function parse_one_obj(s::String, p::Matcher)
    objs = parse_one(s,p)
    @assert length(objs) == 1 "$objs is not a single object"
    objs[1]
end


function compile_circuit_format_lines(lines::Vector{CircuitFormatLine})::Vector{LogicalCircuitNode}
    lin = Vector{CircuitNode}()
    node_cache = Dict{UInt32,CircuitNode}()

    #  literal cache is responsible for making leaf nodes unique and adding them to lin
    lit_cache = Dict{Int32,LeafNode}()
    literal_node(l::Lit) = get!(lit_cache, l) do
        leaf = (l>0 ? PosLeafNode(l) : NegLeafNode(-l)) #it's important for l to be a signed int!'
        push!(lin,leaf)
        leaf
    end

    compile(::Union{HeaderLine,CommentLine}) = () # do nothing
    function compile(ln::PosLiteralLine)
        node_cache[ln.node_id] = literal_node(var2lit(ln.variable))
    end
    function compile(ln::NegLiteralLine)
        node_cache[ln.node_id] = literal_node(-var2lit(ln.variable))
    end
    function compile(ln::LiteralLine)
        node_cache[ln.node_id] = literal_node(ln.literal)
    end
    function compile(ln::TrueLeafLine)
        n = ⋁Node([literal_node(var2lit(ln.variable)), literal_node(-var2lit(ln.variable))])
        push!(lin,n)
        node_cache[ln.node_id] = n
    end
    function compile_elements(e::ElementTuple)
        n = ⋀Node([node_cache[e.prime_id],node_cache[e.sub_id]])
        push!(lin,n)
        n
    end
    function compile(ln::DecisionLine)
        n = ⋁Node(map(compile_elements, ln.elements))
        push!(lin,n)
        node_cache[ln.node_id] = n
    end
    function compile(ln::BiasLine)
        n = ⋁Node([lin[end]])
        push!(lin,n)
        node_cache[ln.node_id] = n
    end

    for ln in lines
        compile(ln)
    end

    lin
end

function load_circuit(file::String)::Vector{LogicalCircuitNode}
    if endswith(file,".circuit")
        load_lc_circuit(file)
    elseif endswith(file,".psdd")
        load_psdd_circuit(file)
    end
end


#####################
# parser of logistic circuit file format
#####################


function parse_lc_decision_line_fast(ln::String)::LCDecisionLine
    head::SubString, tail::SubString = split(ln,'(',limit=2)
    head_tokens = split(head)
    head_ints::Vector{UInt32} = map(x->parse(UInt32,x),head_tokens[2:4])
    elems_str::String = "("*tail
    elems = Vector{ElementTuple}()
    for x in eachmatch(parens::Regex, elems_str)
        tokens = split(x[1], limit=3)
        weights::Vector{Float32} = map(x->parse(Float32,x), split(tokens[3]))
        elem = LCElementTuple(parse(UInt32,tokens[1]), parse(UInt32,tokens[2]), weights)
        push!(elems,elem)
    end
    LCDecisionLine(head_ints[1],head_ints[2],head_ints[3],elems)
end

function parse_true_literal_line_fast(ln::String)::PosLiteralLine
    tokens = split(ln)
    head_ints = map(x->parse(UInt32,x),tokens[2:4])
    weights = map(x->parse(Float32,x), tokens[5:end])
    PosLiteralLine(head_ints[1],head_ints[2],head_ints[3],weights)
end

function parse_false_literal_line_fast(ln::String)::NegLiteralLine
    tokens = split(ln)
    head_ints = map(x->parse(UInt32,x),tokens[2:4])
    weights = map(x->parse(Float32,x), tokens[5:end])
    NegLiteralLine(head_ints[1],head_ints[2],head_ints[3],weights)
end

parse_comment_line_fast(ln::String) = CommentLine(lstrip(chop(ln, head = 1, tail = 0)))

function parse_lc_file(file::String)::Vector{CircuitFormatLine}
    # following one-liner is correct (after dropping Eos()) but much much slower
    # parse_all_nocache(LineSource(open(file)), PlusList!(line,Drop(Pattern(r"\n*"))))
    q = Vector{CircuitFormatLine}()
    open(file) do file # buffered IO does not seem to speed this up
        for ln in eachline(file)
            # hardcode some simpler parsers to speed things up
            if ln[1] == 'D'
                push!(q, parse_lc_decision_line_fast(ln))
            elseif ln[1] == 'T'
                push!(q, parse_true_literal_line_fast(ln))
            elseif ln[1] == 'F'
                push!(q, parse_false_literal_line_fast(ln))
            elseif ln[1] == 'c'
                push!(q, parse_comment_line_fast(ln))
            else
                push!(q, parse_one_obj(ln, circuit_matchers.lc_line))
            end
        end
    end
    q
end

load_lc_circuit(file::String)::Vector{LogicalCircuitNode} = compile_circuit_format_lines(parse_lc_file(file))

#####################
# parser for PSDD circuit format
#####################

function parse_psdd_decision_line_fast(ln::String)::PSDDDecisionLine
    tokens = split(ln)
    head_ints::Vector{UInt32} = map(x->parse(UInt32,x),tokens[2:4])
    elems = Vector{ElementTuple}()
    for (p,s,w) in Iterators.partition(tokens[5:end],3)
        prime = parse(UInt32,p)
        sub = parse(UInt32,s)
        weight = parse(Float32,w)
        elem = PSDDElementTuple(prime, sub, weight)
        push!(elems,elem)
    end
    PSDDDecisionLine(head_ints[1],head_ints[2],head_ints[3],elems)
end

function parse_true_leaf_line_fast(ln::String)::TrueLeafLine
    tokens = split(ln)
    @assert length(tokens)==5
    head_ints = map(x->parse(UInt32,x),tokens[2:4])
    weight = parse(Float32,tokens[5])
    TrueLeafLine(head_ints[1],head_ints[2],head_ints[3],weight)
end

function parse_literal_line_fast(ln::String)::LiteralLine
    tokens = split(ln)
    @assert length(tokens)==4
    head_ints = map(x->parse(UInt32,x),tokens[2:3])
    LiteralLine(head_ints[1],head_ints[2],parse(Int32,tokens[4]))
end

function parse_psdd_file(file::String)::Vector{CircuitFormatLine}
    q = Vector{CircuitFormatLine}()
    open(file) do file # buffered IO does not seem to speed this up
        for ln in eachline(file)
            # hardcode some simpler parsers to speed things up
            if ln[1] == 'D'
                push!(q, parse_psdd_decision_line_fast(ln))
            elseif ln[1] == 'T'
                push!(q, parse_true_leaf_line_fast(ln))
            elseif ln[1] == 'L'
                push!(q, parse_literal_line_fast(ln))
            elseif ln[1] == 'c'
                push!(q, parse_comment_line_fast(ln))
            elseif startswith(ln,"psdd")
                push!(q, HeaderLine())
            else
                error("Don't know how to parse PSDD file format line $ln")
            end
        end
    end
    q
end

load_psdd_circuit(file::String)::Vector{LogicalCircuitNode} = compile_circuit_format_lines(parse_psdd_file(file))

#### Temporary, To refactor later

function compile_prob_circuit_format_lines(lines::Vector{CircuitFormatLine})::Vector{ProbCircuitNode}
    lin = Vector{ProbCircuitNode}()
    node_cache = Dict{UInt32, CircuitNode}()
    prob_cache = ProbCache()

    #  literal cache is responsible for making leaf nodes unique and adding them to lin
    lit_cache = Dict{Int32, LeafNode}()
    literal_node(l::Lit) = get!(lit_cache, l) do
        leaf = (l>0 ? PosLeafNode(l) : NegLeafNode(-l)) #it's important for l to be a signed int!'
        prob_leaf = (l > 0 ? ProbPosLeaf(leaf) : ProbNegLeaf(leaf))
        push!(lin, prob_leaf)
        leaf
    end

    compile(::Union{HeaderLine,CommentLine}) = () # do nothing
    function compile(ln::PosLiteralLine)
        node_cache[ln.node_id] = literal_node(var2lit(ln.variable))
    end
    function compile(ln::NegLiteralLine)
        node_cache[ln.node_id] = literal_node(-var2lit(ln.variable))
    end
    function compile(ln::LiteralLine)
        node_cache[ln.node_id] = literal_node(ln.literal)
    end
    function compile(ln::TrueLeafLine)
        temp = ⋁Node([literal_node(var2lit(ln.variable)), literal_node(-var2lit(ln.variable))])
        n = ProbCircuitNode(
            temp,
            prob_cache
        )
        push!(lin,n)
        node_cache[ln.node_id] = temp
    end
    function compile_elements(e::ElementTuple)
        temp = ⋀Node([node_cache[e.prime_id],node_cache[e.sub_id]])
        n = ProbCircuitNode(
            temp,
            prob_cache
        )
        push!(lin,n)
        temp
    end
    function compile(ln::DecisionLine)
        temp = ⋁Node(map(compile_elements, ln.elements))
        n = ProbCircuitNode(
            temp,
            prob_cache
        )
        n.log_thetas = [x.weight for x in ln.elements]
        push!(lin,n)
        node_cache[ln.node_id] = temp
    end
    function compile(ln::BiasLine)
        temp = ⋁Node([lin[end]])
        n = ProbCircuitNode(
            temp,
            prob_cache
        )
        push!(lin,n)
        node_cache[ln.node_id] = temp
    end

    for ln in lines
        compile(ln)
    end

    lin
end


load_psdd_prob_circuit(file::String)::Vector{ProbCircuitNode} = compile_prob_circuit_format_lines(parse_psdd_file(file))
