#####################
# general parser infrastructure for circuits
#####################

# The following library works correctly but is orders of magnitude too slow.
# using ParserCombinator
# hardcode some simpler parsers to speed things up

"""
Load a logical circuit from file. Depending on format will load different circuit types.

For example, ".psdd" is for PSDD files, and ".circuit" is for Logistic Circuit files.
"""
function load_logical_circuit(file::String)::UnstructLogicalCircuit△
    compile_lines_logical(parse_circuit_file(file))
end

function load_struct_logical_circuit(circuit_file::String, vtree_file::String)
    circuit_lines = parse_circuit_file(circuit_file)
    vtree_lines = parse_vtree_file(vtree_file)
    compile_lines_struct_logical_vtree(circuit_lines, vtree_lines)
end


"""
Load a probabilistic circuit from file. 

For now only ".psdd" PSDD files are supported.
"""
function load_prob_circuit(file::String)::ProbCircuit△
    @assert endswith(file,".psdd")
    compile_lines_prob(parse_psdd_file(file))
end

#####################
# parse based on file extension
#####################

function parse_circuit_file(file::String)::Vector{CircuitFormatLine}
    if endswith(file,".circuit")
        parse_lc_file(file)
    elseif endswith(file,".psdd")
        parse_psdd_file(file)
    else
        throw("Cannot parse this file type as a logical circuit: $file")
    end
end

#####################
# parser of logistic circuit file format
#####################

const parens = r"\(([^\)]+)\)"

function parse_lc_decision_line(ln::String)::LCDecisionLine
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

function parse_true_literal_line(ln::String)::WeightedPosLiteralLine
    @assert startswith(ln, "T")
    tokens = split(ln)
    head_ints = map(x->parse(UInt32,x),tokens[2:4])
    weights = map(x->parse(Float32,x), tokens[5:end])
    WeightedPosLiteralLine(head_ints[1],head_ints[2],head_ints[3],weights)
end

function parse_false_literal_line(ln::String)::WeightedNegLiteralLine
    @assert startswith(ln, "F")
    tokens = split(ln)
    head_ints = map(x->parse(UInt32,x),tokens[2:4])
    weights = map(x->parse(Float32,x), tokens[5:end])
    WeightedNegLiteralLine(head_ints[1],head_ints[2],head_ints[3],weights)
end

function parse_comment_line(ln::String)
    @assert startswith(ln, "c")
    CommentLine(lstrip(chop(ln, head = 1, tail = 0)))
end

function parse_lc_header_line(ln::String)
    @assert (ln == "Logistic Circuit") || (ln == "Logisitic Circuit")
    HeaderLine()
end

function parse_bias_line(ln::String)::BiasLine
    @assert startswith(ln, "B")
    tokens = split(ln)
    weights = map(x->parse(Float32,x), tokens[2:end])
    BiasLine(weights)
end

function parse_lc_file(file::String)::Vector{CircuitFormatLine}
    q = Vector{CircuitFormatLine}()
    open(file) do file # buffered IO does not seem to speed this up
        for ln in eachline(file)
            @assert !isempty(ln)
            if ln[1] == 'D'
                push!(q, parse_lc_decision_line(ln))
            elseif ln[1] == 'T'
                push!(q, parse_true_literal_line(ln))
            elseif ln[1] == 'F'
                push!(q, parse_false_literal_line(ln))
            elseif ln[1] == 'c'
                push!(q, parse_comment_line(ln))
            elseif ln[1] == 'L'
                push!(q, parse_lc_header_line(ln))
            elseif ln[1] == 'B'
                push!(q, parse_bias_line(ln))
            else
                error("Don't know how to parse LC file format line $ln")
            end
        end
    end
    q
end


#####################
# parser for PSDD circuit file format
#####################

function parse_psdd_decision_line(ln::String)::PSDDDecisionLine
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

function parse_true_leaf_line(ln::String)::TrueLeafLine
    tokens = split(ln)
    @assert length(tokens)==5
    head_ints = map(x->parse(UInt32,x),tokens[2:4])
    weight = parse(Float32,tokens[5])
    TrueLeafLine(head_ints[1],head_ints[2],head_ints[3],weight)
end

function parse_literal_line(ln::String)::LiteralLine
    tokens = split(ln)
    @assert length(tokens)==4 "line has too many tokens: $ln"
    head_ints = map(x->parse(UInt32,x),tokens[2:3])
    LiteralLine(head_ints[1],head_ints[2],parse(Int32,tokens[4]))
end

function parse_psdd_file(file::String)::Vector{CircuitFormatLine}
    q = Vector{CircuitFormatLine}()
    open(file) do file # buffered IO does not seem to speed this up
        for ln in eachline(file)
            @assert !isempty(ln)
            if ln[1] == 'D'
                push!(q, parse_psdd_decision_line(ln))
            elseif ln[1] == 'T'
                push!(q, parse_true_leaf_line(ln))
            elseif ln[1] == 'L'
                push!(q, parse_literal_line(ln))
            elseif ln[1] == 'c'
                push!(q, parse_comment_line(ln))
            elseif startswith(ln,"psdd")
                push!(q, HeaderLine())
            else
                error("Don't know how to parse PSDD file format line $ln")
            end
        end
    end
    q
end