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

#####################
# tests
#####################

function run_tests()
    @testset "Circuit file parser tests" begin
       @test parse_one_obj("c blah blablah", circuit_matchers.line) isa CommentLine
       @test parse_one_obj("c", circuit_matchers.line) isa CommentLine
       @test parse_one_obj("c    blah blablah", circuit_matchers.line) isa CommentLine
       @test parse_one_obj("Logisitic Circuit", circuit_matchers.line) isa HeaderLine
       @test (parse_one_obj("0.11139313932426485 0.5341755009918099 0.4104354811044485 0.2088029562981886 0.38317256253159404 0.21456111303752262 0.33798418436324884 0.7382343563376387 0.5769125897294547 0.13071237914862724"
                            , circuit_matchers.weights) isa Vector{Float32})
       @test (parse_one_obj("T 0 0 1 0.11139313932426485 0.5341755009918099 0.4104354811044485 0.2088029562981886 0.38317256253159404 0.21456111303752262 0.33798418436324884 0.7382343563376387 0.5769125897294547 0.13071237914862724"
                            , circuit_matchers.line) isa PosLiteralLine)
       @test (parse_one_obj("F 1069 490 491 0.6277380017061743 -0.45260459349249044 0.34609986139917703 0.6004763090354547 0.2394524067773312 0.22081649811500942 -0.26666977618500204 0.14544044474614298 0.30372580539872435 0.2192352511676825"
                            , circuit_matchers.line) isa NegLiteralLine)
       @test (parse_one_obj("476 478 0.1842650329258706 -0.2197222399177862 0.1595488827322584 0.6063503985221143 0.17354244510981282 0.44742325847644954 -0.019469279671825785 -0.23255932102618532 -0.12040566796613016 -0.21178600346308427"
                            , circuit_matchers.element) isa LCElementTuple)
       @test (parse_one_obj("D 1799 985 4 (472 474 0.27742886347699697 -0.0894114793745983 0.5298165134268861 0.5827938730880822 0.14116799704274996 0.3970938168763751 0.17798346381236296 0.08917988964843772 -0.05605305315306568 0.1702693902831316) (472 475 0.3833466224435187 0.8445851879217264 -0.3572571803165608 0.1793868357569113 -0.2373580813674068 0.670248227361854 -0.11119443329855791 0.13163431621813051 0.5421030929813475 0.25786192990838014) (473 474 1.0369907390437323 0.44729016983853126 -0.07892427803381961 0.38996680892303803 0.5285038536250287 0.3944289684978373 0.2762655604492141 0.556958084538147 0.2711846681681724 0.39922629776124985) (473 475 0.032883234975809694 -0.02256663542306192 0.6555725013615572 0.5140023339657676 0.11841852634121926 0.14907399101146324 -0.22404529652178906 -0.11976212824115842 -0.15206954052616856 0.0022385109727181413)"
                            , circuit_matchers.line) isa LCDecisionLine)
       @test (parse_one_obj("D 10652 1001 2 (508 511 0.0008337025235152718 -6.048729079142479e-05 0.0012900540050118133 0.006382987897195768 0.00013330176570593142 -0.0034902489721742023 0.003162325487226574 -0.009619185307110537 0.043311151137203116 -0.007194862955461081) (509 511 0.023396488696149225 -0.000729066431265012 6.551173017401332e-06 0.05715185398005281 -0.008310854435718613 -0.003834142193742804 -0.005833871820252338 -0.05352747769146413 0.010573950714222884 0.03262423061844396)"
                            , circuit_matchers.line) isa LCDecisionLine)
       @test (parse_one_obj("B -0.6090213458520287 0.10061233805363132 -0.44510731039287776 -0.4536824618763301 -0.738392695523771 -0.5610245232140584 -0.4586543592164493 -0.07962059343551083 -0.2582953135054242 -0.03257926010007175"
                            , circuit_matchers.line) isa BiasLine)
       @test (parse_file("../circuits/mnist-large.circuit") isa Vector{CircuitFormatLine})
    end
end
