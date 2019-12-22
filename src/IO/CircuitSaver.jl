using Printf

import Base.copy
import LogicCircuits.IO: SDDElement, PSDDElement

# Saving psdd

#####################
# decompile for nodes
#####################


# decompile for psdd
decompile(n::ProbLiteral, node2id, vtree2id)::UnweightedLiteralLine = 
    UnweightedLiteralLine(node2id[n], vtree2id[n.origin.vtree], literal(n), true)

make_element(n::Prob⋀, w::AbstractFloat, node2id) = 
    PSDDElement(node2id[n.children[1]],  node2id[n.children[2]], w)

is_true_node(n)::Bool = 
    GateType(n) isa ⋁ && num_children(n) == 2 && GateType(children(n)[1]) isa LiteralLeaf && GateType(children(n)[2]) isa LiteralLeaf && 
    positive(children(n)[1]) && negative(children(n)[2])

function decompile(n::Prob⋁, node2id, vtree2id)::Union{WeightedNamedConstantLine, DecisionLine{PSDDElement}} 
    if is_true_node(n)
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
    outnodes = filter(n -> !(GateType(n) isa ⋀), ln)
    sizehint!(node2id, length(outnodes))
    index = ID(0) # node id start from 0
    for n in outnodes
        node2id[n] = index
        index += ID(1)
    end
    node2id
end

function get_vtree2id(ln::PlainVtree):: Dict{PlainVtreeNode, ID}
    vtree2id = Dict{PlainVtreeNode, ID}()
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

function save_psdd_file(name::String, ln::ProbΔ, vtree::PlainVtree)
    @assert ln[end].origin isa StructLogicalΔNode "PSDD should decorate on StructLogicalΔ"
    @assert endswith(name, ".psdd")

    node2id = get_node2id(ln, ProbΔNode)
    vtree2id = get_vtree2id(vtree)
    formatlines = Vector{CircuitFormatLine}()
    for n in filter(n -> !(GateType(n) isa ⋀), ln)
        push!(formatlines, decompile(n, node2id, vtree2id))
    end
    open(name, "w") do f
        println(f, save_psdd_comment_line())
        println(f, "psdd " * string(length(ln)))
    end
    save_lines(name, formatlines)
end

save_sdd_file(name::String, ln::ProbΔ, vtree::PlainVtree) = 
    save_sdd_file(name, origin(ln), vtree)

function save_sdd_file(name::String, ln::StructLogicalΔ, vtree::PlainVtree)
    @assert endswith(name, ".sdd")
    node2id = get_node2id(ln, StructLogicalΔNode)
    vtree2id = get_vtree2id(vtree)
    formatlines = Vector{CircuitFormatLine}()
    for n in filter(n -> !(GateType(n) isa ⋀), ln)
        push!(formatlines, decompile(n, node2id, vtree2id))
    end

    open(name, "w") do f
        println(f, save_sdd_comment_line())
        println(f, "sdd " * string(length(ln)))
    end

    save_lines(name, formatlines)
end

#TODO replace by dispatch based on second argument type
function save_circuit(name::String, ln, vtree=nothing)
    if endswith(name, ".sdd")
        save_sdd_file(name, ln, vtree)
    elseif endswith(name, ".psdd")
        save_psdd_file(name, ln, vtree)
    elseif endswith(name, ".circuit")
        save_lc_file(name, ln)
    else
        throw("Cannot save this file type as a circuit: $name")
    end
    nothing
end

# TODO 
# function save_lc_file(name::String, ln)
# end
# Saving Logistic Circuits

# Save as .dot
"Rank nodes in the same layer left to right"
function get_nodes_level(circuit::ProbΔ)
    levels = Vector{Vector{ProbΔNode}}()
    current = Vector{ProbΔNode}()
    next = Vector{ProbΔNode}()

    push!(next, circuit[end])
    push!(levels, Base.copy(next))
    while !isempty(next)
        current, next = next, current
        while !isempty(current)
            n = popfirst!(current)
            if n isa ProbInnerNode
                for c in children(n)
                    if !(c in next) push!(next, c); end
                end
            end
        end
        push!(levels, Base.copy(next))
    end

    return levels
end

"Save prob circuits to .dot file"
function save_as_dot(circuit::ProbΔ, file::String)

    node_cache = Dict{ProbΔNode, Int64}()
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
        elseif n isa ProbLiteral && positive(n)
            write(f, "$(node_cache[n]) [label=\"+$(variable(n.origin))\"]\n")
        elseif n isa ProbLiteral && negative(n)
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