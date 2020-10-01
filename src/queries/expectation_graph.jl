export UpExpFlow, ExpFlowCircuit, exp_pass_up, ExpectationUpward

#####################
# Expectation Flow circuits
# For use of algorithms depending on pairs of nodes of two circuits
#####################

"A expectation circuit node that has pair of origins of type PC and type LC"
abstract type ExpFlowNode{F} end

const ExpFlowCircuit{O} = Vector{<:ExpFlowNode{<:O}}

struct UpExpFlow{F} <: ExpFlowNode{F}
    p_origin::ProbCircuit
    f_origin::LogisticCircuit
    children::Vector{<:ExpFlowNode{<:F}}
    f::F
    fg::F
end

import LogicCircuits: children
children(x::UpExpFlow) = x.children

"""
Expected Prediction of LC w.r.t PC.
This implementation uses the computation graph approach.
"""
function ExpectationUpward(pc::ProbCircuit, lc::LogisticCircuit, data)
    # 1. Get probability of each observation
    log_likelihoods = marginal(pc, data)
    p_observed = exp.( log_likelihoods )
    
    # 2. Expectation w.r.t. P(x_m, x_o)
    exps_flow = exp_pass_up(pc, lc, data)
    results_unnormalized = exps_flow[end].fg

    # 3. Expectation w.r.t P(x_m | x_o)
    results = transpose(results_unnormalized) ./ p_observed

    # 4. Add Bias terms
    biases = lc.thetas
    results .+= biases
    
    results, exps_flow
end


"""
Construct a upward expectation flow circuit from a given pair of PC and LC circuits
Note that its assuming the two circuits share the same vtree
"""
function ExpFlowCircuit(pc::ProbCircuit, lc::LogisticCircuit, batch_size::Int, ::Type{El}) where El
    F = Array{El, 2}
    fmem = () -> zeros(1, batch_size) #Vector{El}(undef, batch_size)  #init_array(El, batch_size) # note: fmem's return type will determine type of all UpFlows in the circuit (should be El)
    fgmem = () -> zeros(num_classes(lc), batch_size)

    root_pc = pc
    root_lc = children(lc)[1]
    
    cache = Dict{Pair{Node, Node}, ExpFlowNode}()
    sizehint!(cache, (num_nodes(pc) + num_nodes(lc))*4÷3)
    expFlowCircuit = Vector{ExpFlowNode}()

    function ExpflowTraverse(n::PlainSumNode, m::Logistic⋁Node) 
        get!(cache, Pair(n, m)) do
            ch = [ ExpflowTraverse(i, j) for i in children(n) for j in children(m)]
            node = UpExpFlow{F}(n, m, ch, fmem(), fgmem())
            push!(expFlowCircuit, node)
            return node
        end
    end
    function ExpflowTraverse(n::PlainMulNode, m::Logistic⋀Node) 
        get!(cache, Pair(n, m)) do
            ch = [ ExpflowTraverse(z[1], z[2]) for z in zip(children(n), children(m)) ]
            node = UpExpFlow{F}(n, m, ch, fmem(), fgmem())
            push!(expFlowCircuit, node)
            return node
        end
    end
    function ExpflowTraverse(n::PlainProbLiteralNode, m::Logistic⋁Node) 
        get!(cache, Pair(n, m)) do
            ch = Vector{ExpFlowNode{F}}() # TODO
            node = UpExpFlow{F}(n, m, ch, fmem(), fgmem())
            push!(expFlowCircuit, node)
            return node
        end
    end
    function ExpflowTraverse(n::PlainProbLiteralNode, m::LogisticLiteral) 
        get!(cache, Pair(n, m)) do
            ch = Vector{ExpFlowNode{F}}() # TODO
            node = UpExpFlow{F}(n, m, ch, fmem(), fgmem())
            push!(expFlowCircuit, node)
            return node
        end
    end

    ExpflowTraverse(root_pc, root_lc)
    expFlowCircuit
end

function exp_pass_up(pc::ProbCircuit, lc::LogisticCircuit, data)
    expFlowCircuit = ExpFlowCircuit(pc, lc, num_examples(data), Float64);
    for n in expFlowCircuit
        exp_pass_up_node(n, data)
    end 
    expFlowCircuit
end

function exp_pass_up(fc::ExpFlowCircuit, data)
    #TODO write resize_flows similar to flow_circuits
    #     and give as input the expFlowCircuit instead
    #expFlowCircuit = ExpFlowCircuit(pc, lc, num_examples(data), Float64);
    for n in fc
        exp_pass_up_node(n, data)
    end
end

function exp_pass_up_node(node::ExpFlowNode{E}, data) where E
    pType = typeof(node.p_origin)
    fType = typeof(node.f_origin)

    if node.p_origin isa PlainSumNode && node.f_origin isa Logistic⋁Node
        #todo this ordering might be different than the ExpFlowNode children
        pthetas = [exp(node.p_origin.log_probs[i])
                    for i in 1:length(children(node.p_origin)) for j in 1:length(children(node.f_origin))]
        fthetas = [node.f_origin.thetas[j,:] # only taking the first class for now
            for i in 1:length(node.p_origin.children) for j in 1:length(node.f_origin.children)]

        node.f .= 0.0
        node.fg .= 0.0
        for z = 1:length(children(node))
            node.f  .+= pthetas[z] .* children(node)[z].f
            node.fg .+= (pthetas[z] .* fthetas[z]) .* children(node)[z].f
            node.fg .+= pthetas[z] .* children(node)[z].fg
        end
    elseif node.p_origin isa PlainMulNode && node.f_origin isa Logistic⋀Node
        node.f .= children(node)[1].f .* children(node)[2].f # assume 2 children
        node.fg .= (children(node)[1].f .* children(node)[2].fg) .+
                   (children(node)[2].f .* children(node)[1].fg)

    elseif node.p_origin isa PlainProbLiteralNode 
        if node.f_origin isa Logistic⋁Node
            m = children(node.f_origin)[1]
        elseif node.f_origin isa LogisticLiteral
            m = node.f_origin
        else
            error("Invalid Types of pairs {$pType} - {$fType}")
        end

        var = variable(m)
        X = data
        if ispositive(node.p_origin) && ispositive(m)
            node.f[:, X[:, var] .!= 0 ] .= 1.0 # positive and missing observations
            node.f[:, X[:, var] .== 0 ] .= 0.0
        elseif isnegative(node.p_origin) && isnegative(m)
            node.f[:, X[:, var] .!= 1 ] .= 1.0 # negative and missing observations
            node.f[:, X[:, var] .== 1 ] .= 0.0
        else
            node.f .= 0.0
        end

        if node.f_origin isa Logistic⋁Node
            node.fg .= node.f .* transpose(node.f_origin.thetas)
        else
            node.fg .= 0.0
        end

    else
        error("Invalid Types of pairs {$pType} - {$fType}")
    end

end