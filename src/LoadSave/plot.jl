export DiGraph, plot
using LightGraphs
using TikzGraphs

import LightGraphs: DiGraph

function DiGraph(pc::ProbCircuit)
    edge_labels = Dict()
    label = label = Vector{String}(undef, num_nodes(pc))

    add_label!(g, dict, n::ProbCircuit) = begin
        label[dict[n]] = 
        if isliteralgate(n) "$(literal(n))"
        elseif ismul(n) "*"
        else "+"
        end
    end

    on_edge(g, id_dict, n, c, n_id, c_id) = noop
    on_edge(g, id_dict, n::Union{PlainSumNode, StructSumNode}, c, n_id, c_id) = begin
        edge_labels[(n_id, c_id)] = begin
            i = findall(x -> x === c, children(n))[1]
            "$(exp(n.log_probs[i]))"
        end
    end
    g, _ = LogicCircuits.LoadSave.DiGraph(pc;on_edge=on_edge, on_var=add_label!)
    g, label, edge_labels
end

import TikzGraphs: plot
plot(pc::ProbCircuit) = begin
    g, label, edge_labels = DiGraph(pc)
    TikzGraphs.plot(g, label, edge_labels=edge_labels)
end