
using MetaGraphs: MetaDiGraph, set_prop!, props

#####################
# general parser infrastructure for circuits
#####################

# The `ParserCombinator` library works correctly but is orders of magnitude too slow.
# Instead here we hardcode some simpler parsers to speed things up

"""
Load a logical circuit from file.
Support file formats:
 * ".sdd" for SDD files
 * ".psdd" for PSDD files
 * ".circuit" for Logistic Circuit files
"""
function load_logical_circuit(file::String)::UnstLogicalΔ
    compile_logical(parse_circuit_file(file))
end

"""
Load a smooth logical circuit from file.
Support file formats:
 * ".psdd" for PSDD files
 * ".circuit" for Logistic Circuit files
"""
function load_smooth_logical_circuit(file::String)::UnstLogicalΔ
    compile_smooth_logical(parse_circuit_file(file))
end

"""
Load a smooth structured logical circuit from file.
Support circuit file formats:
 * ".psdd" for PSDD files
 * ".circuit" for Logistic Circuit files
Supported vtree file formats:
 * ".vtree" for VTree files
"""
function load_struct_smooth_logical_circuit(circuit_file::String, vtree_file::String)::Tuple{StructLogicalCircuit{PlainVtreeNode},PlainVtree}
    circuit_lines = parse_circuit_file(circuit_file)
    vtree_lines = parse_vtree_file(vtree_file)
    compile_smooth_struct_logical(circuit_lines, vtree_lines)
end

"""
Load a probabilistic circuit from file.
Support circuit file formats:
 * ".psdd" for PSDD files
 """
function load_prob_circuit(file::String)::ProbΔ
    @assert endswith(file,".psdd")
    compile_prob(parse_psdd_file(file))
end

"""
Load a structured probabilistic circuit from file.
Support circuit file formats:
 * ".psdd" for PSDD files
Supported vtree file formats:
 * ".vtree" for VTree files
"""
function load_struct_prob_circuit(circuit_file::String, vtree_file::String)::Tuple{ProbΔ,PlainVtree}
    @assert endswith(circuit_file,".psdd")
    circuit_lines = parse_circuit_file(circuit_file)
    vtree_lines = parse_vtree_file(vtree_file)
    compile_struct_prob(circuit_lines, vtree_lines)
end


function load_logistic_circuit(circuit_file::String, classes::Int)::LogisticΔ
    @assert endswith(circuit_file,".circuit")
    circuit_lines = parse_circuit_file(circuit_file)
    compile_logistic(circuit_lines, classes)
end


#####################
# parse based on file extension
#####################

function parse_circuit_file(file::String)::CircuitFormatLines
    if endswith(file,".circuit")
        parse_lc_file(file)
    elseif endswith(file,".psdd")
        parse_psdd_file(file)
    elseif endswith(file,".sdd")
        parse_sdd_file(file)
    else
        throw("Cannot parse this file type as a circuit: $file")
    end
end

#####################
# parser of logistic circuit file format
#####################

const parens = r"\(([^\)]+)\)"

function parse_lc_decision_line(ln::String)::DecisionLine{LCElement}
    @assert startswith(ln, "D")
    head::SubString, tail::SubString = split(ln,'(',limit=2)
    head_tokens = split(head)
    head_ints::Vector{UInt32} = map(x->parse(UInt32,x),head_tokens[2:4])
    elems_str::String = "("*tail
    elems = Vector{LCElement}()
    for x in eachmatch(parens::Regex, elems_str)
        tokens = split(x[1], limit=3)
        weights::Vector{Float64} = map(x->parse(Float64,x), split(tokens[3]))
        elem = LCElement(parse(UInt32,tokens[1]), parse(UInt32,tokens[2]), weights)
        push!(elems,elem)
    end
    DecisionLine(head_ints[1],head_ints[2],head_ints[3],elems)
end

function parse_lc_literal_line(ln::String)::WeightedLiteralLine
    @assert startswith(ln, "T") || startswith(ln, "F")
    tokens = split(ln)
    head_ints = map(x->parse(UInt32,x),tokens[2:4])
    weights = map(x->parse(Float64,x), tokens[5:end])
    lit = var2lit(head_ints[3])
    if startswith(ln, "F")
       lit = -lit # negative literal
    end
    WeightedLiteralLine(head_ints[1],head_ints[2],lit,true,weights)
end

function parse_comment_line(ln::String)
    @assert startswith(ln, "c")
    CircuitCommentLine(lstrip(chop(ln, head = 1, tail = 0)))
end

function parse_lc_header_line(ln::String)
    @assert (ln == "Logistic Circuit") || (ln == "Logisitic Circuit")
    CircuitHeaderLine()
end

function parse_bias_line(ln::String)::BiasLine
    @assert startswith(ln, "B")
    tokens = split(ln)
    weights = map(x->parse(Float64,x), tokens[2:end])
    BiasLine(weights)
end

function parse_lc_file(file::String)::CircuitFormatLines
    q = Vector{CircuitFormatLine}()
    open(file) do file # buffered IO does not seem to speed this up
        for ln in eachline(file)
            @assert !isempty(ln)
            if ln[1] == 'D'
                push!(q, parse_lc_decision_line(ln))
            elseif ln[1] == 'T' || ln[1] == 'F'
                push!(q, parse_lc_literal_line(ln))
            elseif ln[1] == 'c'
                push!(q, parse_comment_line(ln))
            elseif ln[1] == 'L'
                push!(q, parse_lc_header_line(ln))
            elseif ln[1] == 'B'
                push!(q, parse_bias_line(ln))
            else
                error("Cannot parse logistic circuit file format line $ln")
            end
        end
    end
    q
end


#####################
# parser for PSDD circuit file format
#####################

function parse_psdd_decision_line(ln::String)::DecisionLine{PSDDElement}
    @assert startswith(ln, "D")
    tokens = split(ln)
    head_ints::Vector{UInt32} = map(x->parse(UInt32,x),tokens[2:4])
    elems = Vector{PSDDElement}()
    for (p,s,w) in Iterators.partition(tokens[5:end],3)
        prime = parse(UInt32,p)
        sub = parse(UInt32,s)
        weight = parse(Float64,w)
        elem = PSDDElement(prime, sub, weight)
        push!(elems,elem)
    end
    DecisionLine(head_ints[1],head_ints[2],head_ints[3],elems)
end

function parse_psdd_true_leaf_line(ln::String)::WeightedNamedConstantLine
    @assert startswith(ln, "T")
    tokens = split(ln)
    @assert length(tokens)==5
    head_ints = map(x->parse(UInt32,x),tokens[2:4])
    weight = parse(Float64,tokens[5])
    WeightedNamedConstantLine(head_ints[1],head_ints[2],head_ints[3],weight)
end

function parse_literal_line(ln::String, normalized::Bool)::UnweightedLiteralLine
    @assert startswith(ln, "L")
    tokens = split(ln)
    @assert length(tokens)==4 "line has too many tokens: $ln"
    head_ints = map(x->parse(UInt32,x),tokens[2:3])
    lit = parse(Int32,tokens[4])
    UnweightedLiteralLine(head_ints[1],head_ints[2],lit,normalized)
end

function parse_psdd_file(file::String)::CircuitFormatLines
    q = Vector{CircuitFormatLine}()
    open(file) do file # buffered IO does not seem to speed this up
        for ln in eachline(file)
            @assert !isempty(ln)
            if ln[1] == 'D'
                push!(q, parse_psdd_decision_line(ln))
            elseif ln[1] == 'T'
                push!(q, parse_psdd_true_leaf_line(ln))
            elseif ln[1] == 'L'
                push!(q, parse_literal_line(ln, true))
            elseif ln[1] == 'c'
                push!(q, parse_comment_line(ln))
            elseif startswith(ln,"psdd")
                push!(q, CircuitHeaderLine())
            else
                error("Cannot parse PSDD file format line $ln")
            end
        end
    end
    q
end

#####################
# parser for SDD circuit file format
#####################

function parse_sdd_decision_line(ln::String)::DecisionLine{SDDElement}
    @assert startswith(ln, "D")
    tokens = split(ln)
    head_ints::Vector{UInt32} = map(x->parse(UInt32,x),tokens[2:4])
    elems = Vector{SDDElement}()
    for (p,s) in Iterators.partition(tokens[5:end],2)
        prime = parse(UInt32,p)
        sub = parse(UInt32,s)
        elem = SDDElement(prime, sub)
        push!(elems,elem)
    end
    DecisionLine(head_ints[1],head_ints[2],head_ints[3],elems)
end

function parse_sdd_constant_leaf_line(ln::String)::AnonymousConstantLine
    @assert startswith(ln, "T") || startswith(ln, "F")
    tokens = split(ln)
    @assert length(tokens)==2
    AnonymousConstantLine(parse(UInt32,tokens[2]), startswith(ln, "T"), false)
end

function parse_sdd_file(file::String)::CircuitFormatLines
    q = Vector{CircuitFormatLine}()
    open(file) do file # buffered IO does not seem to speed this up
        for ln in eachline(file)
            @assert !isempty(ln)
            if ln[1] == 'D'
                push!(q, parse_sdd_decision_line(ln))
            elseif ln[1] == 'T' || ln[1] == 'F'
                push!(q, parse_sdd_constant_leaf_line(ln))
            elseif ln[1] == 'L'
                push!(q, parse_literal_line(ln, false))
            elseif ln[1] == 'c'
                push!(q, parse_comment_line(ln))
            elseif startswith(ln,"sdd")
                push!(q, CircuitHeaderLine())
            else
                error("Cannot parse SDD file format line $ln")
            end
        end
    end
    q
end

#####################
# loader for CNF file format
#####################

"""
Load a CNF as a logical circuit from file.
Supppor file formats:
* ".cnf" for CNF files
"""
function load_cnf(file::String)::UnstLogicalΔ
    @assert endswith(file, ".cnf")

    # linearized circuit nodes
    circuit = Vector{UnstLogicalΔNode}()

    # linearized clauses (disjunctions)
    clauses = Vector{⋁Node}()

    # literal cache is responsible for making leaf literals nodes unique and adding them to `circuit`
    lit_cache = Dict{Lit,LogicalLeafNode}()
    literal_node(l::Lit) = get!(lit_cache, l) do
        leaf = LiteralNode(l)
        push!(circuit,leaf) # also add new leaf to linearized circuit before caller
        leaf
    end

    # record the current clause
    clause = ⋁Node([])

    count_clauses = 0

    open(file) do file

        for ln in eachline(file)
            @assert !isempty(ln)
            if ln[1] == 'c' || startswith(ln, "p cnf")
                # skip comment and header lines
                continue
            else
                tokens = split(ln)
                for token in tokens
                    if !occursin(r"^\s*[-]?[0-9]+\s*$", token)
                        error("Cannot parse CNF file format line $ln")
                    end
                    literal = parse(Lit, token)
                    if literal == 0
                        push!(clauses, clause)
                        push!(circuit, clause)
                        clause = ⋁Node([])
                        count_clauses += 1
                    else
                        push!(clause.children, literal_node(literal))
                    end
                end
            end
        end

        # handle the last clause
        if length(clause.children) > 0
            push!(clauses, clause)
            push!(circuit, clause)
            count_clauses + 1
        end

        # create the root conjunction node
        if length(clauses) > 0
            push!(circuit, ⋀Node(clauses))
        end
    end

    circuit
end

#####################
# loader for DNF file format
#####################

"""
Load a CNF as a logical circuit from file.
Supppor file formats:
* ".cnf" for CNF files
"""
function load_dnf(file::String)::UnstLogicalΔ
    @assert endswith(file, ".dnf")

    # linearized circuit nodes
    circuit = Vector{UnstLogicalΔNode}()

    # linearized clauses (conjunctions)
    clauses = Vector{⋀Node}()

    # literal cache is responsible for making leaf literals nodes unique and adding them to `circuit`
    lit_cache = Dict{Lit,LogicalLeafNode}()
    literal_node(l::Lit) = get!(lit_cache, l) do
        leaf = LiteralNode(l)
        push!(circuit,leaf) # also add new leaf to linearized circuit before caller
        leaf
    end

    # record the current clause
    clause = ⋀Node([])

    open(file) do file

        for ln in eachline(file)
            @assert !isempty(ln)
            if ln[1] == 'c' || startswith(ln, "p dnf")
                # skip comment and header lines
                continue
            else
                tokens = split(ln)
                for token in tokens
                    if !occursin(r"^\s*[-]?[0-9]+\s*$", token)
                        error("Cannot parse CNF file format line $ln")
                    end
                    literal = parse(Lit, token)
                    if literal == 0
                        push!(clauses, clause)
                        push!(circuit, clause)
                        clause = ⋀Node([])
                    else
                        push!(clause.children, literal_node(literal))
                    end
                end
            end
        end

        # handle the last clause
        if length(clause.children) > 0
            push!(clauses, clause)
            push!(circuit, clause)
        end

        # create the root disjunction node
        if length(clauses) > 0
            push!(circuit, ⋁Node(clauses))
        end
    end

    circuit
end


"Parse a clt from given file"
function parse_clt(filename::String)::MetaDiGraph
    f = open(filename)
    n = parse(Int32,readline(f))
    n_root = parse(Int32,readline(f))
    clt = MetaDiGraph(n)
    for i in 1 : n_root
        root, prob = split(readline(f), " ")
        root, prob = parse(Int32, root), parse(Float64, prob)
        set_prop!(clt, root, :parent, 0)
        set_prop!(clt, root, :cpt, Dict(1=>prob,0=>1-prob))
    end

    for i = 1 : n - n_root
        dst, src, prob1, prob0 = split(readline(f), " ")
        dst, src, prob1, prob0 = parse(Int32, dst), parse(Int32, src), parse(Float64, prob1), parse(Float64, prob0)
        add_edge!(clt, src,dst)
        set_prop!(clt, dst, :parent, src)
        set_prop!(clt, dst, :cpt, Dict((1,1)=>prob1, (0,1)=>1-prob1, (1,0)=>prob0, (0,0)=>1-prob0))
    end
    return clt
end
