#=
Circuits:
- Julia version: 1.1.1
- Author: guy
- Date: 2019-06-23
=#

# The following library works correctly but is orders of magnitude too slow.
# using ParserCombinator

using EponymTuples

#####################
# general parser infrastructure for circuits (both LC and PSDD)
#####################

"""
A line in one of the circuit file formats
"""
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

const parens = r"\(([^\)]+)\)"

function compile_circuit_format_lines(lines::Vector{CircuitFormatLine})::Vector{LogicalCircuitNode}
    lin = Vector{CircuitNode}()
    node_cache = Dict{UInt32,CircuitNode}()

    #  literal cache is responsible for making leaf nodes unique and adding them to lin
    lit_cache = Dict{Int32,LogicalLeafNode}()
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

"""
Load a circuit from file. Depending on format will load different circuit types.

For example, ".psdd" is for PSDD files, and ".circuit" is for Logistic Circuit files.
"""
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
    @assert startswith(ln, "D")
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
    @assert startswith(ln, "T")
    tokens = split(ln)
    head_ints = map(x->parse(UInt32,x),tokens[2:4])
    weights = map(x->parse(Float32,x), tokens[5:end])
    PosLiteralLine(head_ints[1],head_ints[2],head_ints[3],weights)
end

function parse_false_literal_line_fast(ln::String)::NegLiteralLine
    @assert startswith(ln, "F")
    tokens = split(ln)
    head_ints = map(x->parse(UInt32,x),tokens[2:4])
    weights = map(x->parse(Float32,x), tokens[5:end])
    NegLiteralLine(head_ints[1],head_ints[2],head_ints[3],weights)
end

function parse_comment_line_fast(ln::String)
    @assert startswith(ln, "c")
    CommentLine(lstrip(chop(ln, head = 1, tail = 0)))
end

function parse_lc_header_line_fast(ln::String)
    @assert (ln == "Logistic Circuit") || (ln == "Logisitic Circuit")
    HeaderLine()
end

function parse_bias_line_fast(ln::String)::BiasLine
    @assert startswith(ln, "B")
    tokens = split(ln)
    weights = map(x->parse(Float32,x), tokens[2:end])
    BiasLine(weights)
end


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
            elseif ln[1] == 'L'
                push!(q, parse_lc_header_line_fast(ln))
            elseif ln[1] == 'B'
                push!(q, parse_bias_line_fast(ln))
            else
                throw("Could not parse line $ln")
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
    lit_cache = Dict{Lit, LogicalLeafNode}()
    literal_node(l::Lit) = get!(lit_cache, l) do
        leaf = (l>0 ? PosLeafNode(l) : NegLeafNode(-l)) #it's important for l to be a signed int!
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
        n.log_thetas .= 0
        n.log_thetas .+= [ln.weight, log(1-exp(ln.weight) + 1e-300) ]
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
        n.log_thetas .= 0
        n.log_thetas .+= [x.weight for x in ln.elements]
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
        n.log_thetas .= 0
        n.log_thetas .+= ln.weights
        node_cache[ln.node_id] = temp
    end

    for ln in lines
        compile(ln)
    end

    # Sanity Check
    for node in lin
        if node isa Prob⋁
            if sum(isnan.(node.log_thetas)) > 0
                throw("There is a NaN in one of the log_thetas")
            end
        end
    end

    lin
end


load_psdd_prob_circuit(file::String)::Vector{ProbCircuitNode} = compile_prob_circuit_format_lines(parse_psdd_file(file))
