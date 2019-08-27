#####################
# general parser infrastructure for circuits
#####################

# The following library works correctly but is orders of magnitude too slow.
# using ParserCombinator

const parens = r"\(([^\)]+)\)"

"""
Load a logical circuit from file. Depending on format will load different circuit types.

For example, ".psdd" is for PSDD files, and ".circuit" is for Logistic Circuit files.
"""
function load_logical_circuit(file::String)::LogicalCircuit△
    if endswith(file,".circuit")
        load_lc_logical_circuit(file)
    elseif endswith(file,".psdd")
        load_psdd_logical_circuit(file)
    else
        throw("Cannot parse this file type as a logical circuit: $file")
    end
end

load_psdd_logical_circuit(file::String)::Vector{LogicalCircuitNode} = compile_lines_logical(parse_psdd_file(file))
load_lc_logical_circuit(file::String)::Vector{LogicalCircuitNode} = compile_lines_logical(parse_lc_file(file))

"""
Load a probabilistic circuit from file. 

For now only ".psdd" PSDD files are supported.
"""
function load_prob_circuit(file::String)::ProbCircuit△
    if endswith(file,".psdd")
        load_psdd_prob_circuit(file)
    else
        throw("Cannot parse this file type as a probabilistic circuit: $file")
    end
end

load_psdd_prob_circuit(file::String)::Vector{ProbCircuitNode} = compile_lines_prob(parse_psdd_file(file))

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