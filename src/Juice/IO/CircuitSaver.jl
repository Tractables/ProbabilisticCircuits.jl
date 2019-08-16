# To add saving code for circuits


# Saving psdd

# Saving Logistic Circuits

# Save as .dot

function save_as_dot(file::String, circuit::ProbCircuit△)

    node_cache = Dict{ProbCircuitNode, Int64}()
    for (i, n) in enumerate(reverse(circuit))
        node_cache[n] = i
    end

    println(length(node_cache))
    f = open(file, "w")
    write(f, "strict graph Circuit {\n")

    for n in reverse(circuit)
        if n isa Prob⋀
            write(f, "$(node_cache[n]) [label=\"*\"] ;\n")
            for c in n.children
                write(f, "$(node_cache[n]) -- $(node_cache[c]) ;\n")
            end
        elseif n isa Prob⋁
            write(f, "$(node_cache[n]) [label=\"+\"] ;\n")
            for (c, p) in zip(n.children, exp.(n.log_thetas))
                #write(f, "$(node_cache[n]) -- $(node_cache[c]) [label=\"$(string(p)[1:3])\"] ;\n")
                write(f, "$(node_cache[n]) -- $(node_cache[c])  ;\n")
            end
        elseif n isa ProbPosLeaf
            write(f, "$(node_cache[n]) [label=\"+$(n.origin.cvar)\"] ;\n")
        elseif n isa ProbNegLeaf
            write(f, "$(node_cache[n]) [label=\"-$(n.origin.cvar)\"] ;\n")
        else
            throw("unknown ProbNode type")
        end
    end
    write(f, "}\n")
    flush(f)
end
