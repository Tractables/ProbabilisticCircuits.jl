
using LinearAlgebra: diagind
"""
Pick the edge with maximum flow
"""
function count_downflow(values::Matrix{UInt64}, flows::Matrix{UInt64}, n::LogicCircuit)
    dec_id = n.data.node_id
    sum(1:size(flows,1)) do i
        count_ones(flows[i, dec_id]) 
    end
end

function count_downflow(values::Matrix{UInt64}, flows::Matrix{UInt64}, n::LogicCircuit, c::LogicCircuit)
    grandpa = n.data.node_id
    prime = c.prime.data.node_id
    sub = c.sub.data.node_id
    edge_count = sum(1:size(flows,1)) do i
        count_ones(values[i, prime] & values[i, sub] & flows[i, grandpa]) 
    end
end

function downflow_all(values::Matrix{UInt64}, flows::Matrix{UInt64}, n::LogicCircuit, c::LogicCircuit)
    grandpa = n.data.node_id
    prime = c.prime.data.node_id
    sub = c.sub.data.node_id
    edge = map(1:size(flows,1)) do i
        digits(Bool, values[i, prime] & values[i, sub] & flows[i, grandpa], base=2, pad=64)
    end
    vcat(edge...)
end

function eFlow(values, flows, candidates::Vector{Tuple{Node, Node}})
    edge2flows = map(candidates) do (or, and)
        count_downflow(values, flows, or, and)
    end
    (max_flow, max_edge_id) = findmax(edge2flows)
    candidates[max_edge_id], max_flow
end

"""
Pick the variable with maximum sum of mutual information
"""
function vMI(values, flows, edge, vars::Vector{Var}, train_x)
    examples_id = downflow_all(values, flows, edge...)[1:num_examples(train_x)]
    sub_matrix = train_x[examples_id, vars]
    (_, mi) = mutual_information(sub_matrix; α=1.0)
    mi[diagind(mi)] .= 0
    scores = dropdims(sum(mi, dims = 1), dims = 1)
    var = vars[argmax(scores)]
    score = maximum(scores)
    var, score
end

"""
Pick the edge randomly
"""
function eRand(candidates::Vector{Tuple{Node, Node}})
    return rand(candidates)
end

"""
Pick the variable randomly
"""
function vRand(vars::Vector{Var})
    lits = collect(Set{Lit}(scope[and]))
    vars =  Var.(intersect(filter(l -> l > 0, lits), - filter(l -> l < 0, lits)))
    return Var(rand(vars))
end

function heuristic_loss(circuit::LogicCircuit, train_x; pick_edge="eFlow", pick_var="vMI")
    candidates, scope = split_candidates(circuit)
    values, flows = satisfies_flows(circuit, train_x)
    if pick_edge == "eFlow"
        edge, flow = eFlow(values, flows, candidates)
    elseif pick_edge == "eRand"
        edge = eRand(candidates)
    else
        error("Heuristics $pick_edge to pick edge is undefined.")
    end

    or, and = edge
    lits = collect(Set{Lit}(scope[and]))
    vars =  Var.(intersect(filter(l -> l > 0, lits), - filter(l -> l < 0, lits)))

    if pick_var == "vMI"
        var, score = vMI(values, flows, edge, vars, train_x)
    elseif pick_var == "vRand"
        var = vRand(vars)
    else
        error("Heuristics $pick_var to pick variable is undefined.")
    end

    return (or, and), var
end

