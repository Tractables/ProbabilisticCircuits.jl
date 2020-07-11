using Printf: @sprintf

import Base.copy
import LogicCircuits.IO: SDDElement, 
    PSDDElement, 
    save_lines, 
    parse_psdd_file, 
    PsddHeaderLine, 
    LcHeaderLine, 
    save_sdd_file, 
    save_as_dot,
    get_nodes_level

# Saving psdd

#####################
# decompile for nodes
#####################

# decompile for psdd
decompile(n::ProbLiteral, node2id, vtree2id)::UnweightedLiteralLine = 
    UnweightedLiteralLine(node2id[n], vtree2id[n.origin.vtree], literal(n), true)

make_element(n::Prob⋀, w::AbstractFloat, node2id) = 
    PSDDElement(node2id[n.children[1]],  node2id[n.children[2]], w)

istrue_node(n)::Bool = 
    GateType(n) isa ⋁Gate && num_children(n) == 2 && GateType(children(n)[1]) isa LiteralGate && GateType(children(n)[2]) isa LiteralGate && 
    ispositive(children(n)[1]) && isnegative(children(n)[2])

function decompile(n::Prob⋁, node2id, vtree2id)::Union{WeightedNamedConstantLine, DecisionLine{PSDDElement}} 
    if istrue_node(n)
        WeightedNamedConstantLine(node2id[n], vtree2id[n.origin.vtree], lit2var(n.children[1].origin.literal), n.log_thetas[1]) # TODO
    else
        DecisionLine(node2id[n], vtree2id[n.origin.vtree], UInt32(num_children(n)), map(x -> make_element(x[1], x[2], node2id), zip(children(n), n.log_thetas)))
    end
end

#####################
# build maping
#####################

function get_node2id(ln::AbstractVector{X}, T::Type)where X #<: T#::Dict{T, ID}
    node2id = Dict{T, ID}()
    outnodes = filter(n -> !(GateType(n) isa ⋀Gate), ln)
    sizehint!(node2id, length(outnodes))
    index = ID(0) # node id start from 0
    for n in outnodes
        node2id[n] = index
        index += ID(1)
    end
    node2id
end

function get_vtree2id(ln::PlainVtree):: Dict{PlainVtree, ID}
    vtree2id = Dict{PlainVtree, ID}()
    sizehint!(vtree2id, length(ln))
    index = ID(0) # vtree id start from 0

    for n in ln
        vtree2id[n] = index
        index += ID(1)
    end
    vtree2id
end

#####################
# saver for circuits
#####################


function psdd_header()
    """
    c ids of psdd nodes start at 0
    c psdd nodes appear bottom-up, children before parents
    c
    c file syntax:
    c psdd count-of-sdd-nodes
    c L id-of-literal-sdd-node id-of-vtree literal
    c T id-of-trueNode-sdd-node id-of-vtree variable log(litProb)
    c D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub log(elementProb)}*
    c
    c File generated by Juice.jl
    c"""
end

function save_psdd_file(name::String, ln::ProbΔ, vtree::PlainVtree)
    # TODO add method isstructured
    @assert ln[end].origin isa StructLogicCircuit "PSDD should decorate on StructLogicΔ"
    @assert endswith(name, ".psdd")
    node2id = get_node2id(ln, ProbNode)
    vtree2id = get_vtree2id(vtree)
    formatlines = Vector{CircuitFormatLine}()
    append!(formatlines, parse_psdd_file(IOBuffer(psdd_header())))
    push!(formatlines, PsddHeaderLine(num_nodes(ln)))
    for n in filter(n -> !(GateType(n) isa ⋀Gate), ln)
        push!(formatlines, decompile(n, node2id, vtree2id))
    end
    save_lines(name, formatlines)
end

function lc_header()
    """
    c variables (from inputs) start from 1
    c ids of logistic circuit nodes start from 0
    c nodes appear bottom-up, children before parents
    c the last line of the file records the bias parameter
    c three types of nodes:
    c	T (terminal nodes that correspond to true literals)
    c	F (terminal nodes that correspond to false literals)
    c	D (OR gates)
    c
    c file syntax:
    c Logistic Circuit
    c T id-of-true-literal-node id-of-vtree variable parameters
    c F id-of-false-literal-node id-of-vtree variable parameters
    c D id-of-or-gate id-of-vtree number-of-elements (id-of-prime id-of-sub parameters)s
    c B bias-parameters
    c
    c File generated by Juice.jl
    c"""
end
    
function save_lc_file(name::String, ln::LogisticΔ, vtree)
    @assert ln[end].origin isa StructLogicCircuit "LC should decorate on StructLogicΔ"
    @assert endswith(name, ".circuit")
    node2id = get_node2id(ln, ProbNode)
    vtree2id = get_vtree2id(vtree)
    formatlines = Vector{CircuitFormatLine}()
    append!(formatlines, parse_lc_file(IOBuffer(lc_header())))
    push!(formatlines, LcHeaderLine())
    for n in filter(n -> !(GateType(n) isa ⋀Gate), ln)
        push!(formatlines, decompile(n, node2id, vtree2id))
    end
    save_lines(name, formatlines)
end

import LogicCircuits.save_circuit # make available for extension

function save_circuit(name::String, circuit, vtree=nothing)
    if endswith(name, ".circuit")
        save_lc_file(name, circuit, vtree)
    elseif endswith(name, ".psdd")
        save_psdd_file(name, circuit, vtree)
    elseif endswith(name, ".sdd")
        save_sdd_file(name, circuit, vtree)
    else
        error("Cannot save circuit to file with this extensions: $name")
    end
end

"Save prob circuit to .dot file"
function save_as_dot(root::ProbNode, file::String)
    return save_as_dot(linearize(root), file)
end

"Save prob circuits to .dot file"
function save_as_dot(circuit::ProbΔ, file::String)
    # TODO (https://github.com/Juice-jl/LogicCircuits.jl/issues/7)
    node_cache = Dict{ProbNode, Int64}()
    for (i, n) in enumerate(circuit)
        node_cache[n] = i
    end

    levels = get_nodes_level(circuit)

    f = open(file, "w")
    write(f,"digraph Circuit {\nsplines=false\nedge[arrowhead=\"none\",fontsize=6]\n")

    for level in levels
        if length(level) > 1
            write(f,"{rank=\"same\";newrank=\"true\";rankdir=\"LR\";")
            rank = ""
            foreach(x->rank*="$(node_cache[x])->",level)
            rank = rank[1:end-2]
            write(f, rank)
            write(f,"[style=invis]}\n")
        end
    end

    for n in reverse(circuit)
        if n isa Prob⋀
            write(f, "$(node_cache[n]) [label=\"*$(node_cache[n])\"]\n")
        elseif n isa Prob⋁
            write(f, "$(node_cache[n]) [label=\"+$(node_cache[n])\"]\n")
        elseif n isa ProbLiteral && ispositive(n)
            write(f, "$(node_cache[n]) [label=\"+$(variable(n.origin))\"]\n")
        elseif n isa ProbLiteral && isnegative(n)
            write(f, "$(node_cache[n]) [label=\"-$(variable(n.origin))\"]\n")
        else
            throw("unknown ProbNode type")
        end
    end

    for n in reverse(circuit)
        if n isa Prob⋀
            for c in n.children
                write(f, "$(node_cache[n]) -> $(node_cache[c])\n")
            end
        elseif n isa Prob⋁
            for (c, p) in zip(n.children, exp.(n.log_thetas))
                prob = @sprintf "%0.1f" p
                write(f, "$(node_cache[n]) -> $(node_cache[c]) [label=\"$prob\"]\n")
            end
        else
        end
    end

    write(f, "}\n")
    flush(f)
    close(f)
end
