using Printf
import Base.copy
# To add saving code for circuits


# Saving psdd

# Saving Logistic Circuits

# Save as .dot
"Rank nodes in the same layer left to right"
function get_nodes_level(circuit::ProbCircuit△)
    levels = Vector{Vector{ProbCircuitNode}}()
    current = Vector{ProbCircuitNode}()
    next = Vector{ProbCircuitNode}()

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
function save_as_dot(circuit::ProbCircuit△, file::String)

    node_cache = Dict{ProbCircuitNode, Int64}()
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