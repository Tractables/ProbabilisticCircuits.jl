########################
# Do not use for now
######################
#####################
# Expectation Flow circuits
# For use of algorithms depending on pairs of nodes of two circuits
#####################

"A expectation circuit node that has pair of origins of type PC and type LC"
abstract type DecoratorNodePair{PC<:Node, LC<:Node} <: Node end

abstract type ExpFlowNode{PC, LC, F} <: DecoratorNodePair{PC, LC} end

const ExpFlowΔ{O} = AbstractVector{<:ExpFlowNode{<:O}}

struct UpExpFlow{PC, LC, F} <: ExpFlowNode{PC, LC, F}
    p_origin::PC
    f_origin::LC
    children::Vector{<:ExpFlowNode{<:PC, <:LC, <:F}}
    f::F
    fg::F
end


"""
Construct a upward expectation flow circuit from a given pair of PC and LC circuits
Note that its assuming the two circuits share the same vtree
"""
function ExpFlowΔ(pc::ProbΔ, lc::LogisticΔ, batch_size::Int, ::Type{El}) where El
    pc_type = grapheltype(pc)
    lc_type = grapheltype(lc)

    F = Array{El, 2}
    fmem = () -> zeros(1, batch_size) #Vector{El}(undef, batch_size)  #init_array(El, batch_size) # note: fmem's return type will determine type of all UpFlows in the circuit (should be El)
    fgmem = () -> zeros(classes(lc[end]), batch_size)

    root_pc = pc[end]
    root_lc = lc[end- 1]
    
    cache = Dict{Pair{Node, Node}, ExpFlowNode}()
    sizehint!(cache, (length(pc) + length(lc))*4÷3)
    expFlowCircuit = Vector{ExpFlowNode}()

    function ExpflowTraverse(n::Prob⋁, m::Logistic⋀Node) 
        get!(cache, Pair(n, m)) do
            children = [ ExpflowTraverse(i, j) for i in n.children for j in m.children]
            node = UpExpFlow{pc_type,lc_type, F}(n, m, children, fmem(), fgmem())
            push!(expFlowCircuit, node)
            return node
        end
    end
    function ExpflowTraverse(n::Prob⋀, m::Logistic⋀Node) 
        get!(cache, Pair(n, m)) do
            children = [ ExpflowTraverse(z[1], z[2]) for z in zip(n.children, m.children) ]
            node = UpExpFlow{pc_type,lc_type, F}(n, m, children, fmem(), fgmem())
            push!(expFlowCircuit, node)
            return node
        end
    end
    function ExpflowTraverse(n::ProbLiteral, m::Logistic⋀Node) 
        get!(cache, Pair(n, m)) do
            children = Vector{ExpFlowNode{pc_type,lc_type, F}}() # TODO
            node = UpExpFlow{pc_type,lc_type, F}(n, m, children, fmem(), fgmem())
            push!(expFlowCircuit, node)
            return node
        end
    end
    function ExpflowTraverse(n::ProbLiteral, m::LogisticLiteral) 
        get!(cache, Pair(n, m)) do
            children = Vector{ExpFlowNode{pc_type,lc_type, F}}() # TODO
            node = UpExpFlow{pc_type,lc_type, F}(n, m, children, fmem(), fgmem())
            push!(expFlowCircuit, node)
            return node
        end
    end

    ExpflowTraverse(root_pc, root_lc)
    expFlowCircuit
end

function exp_pass_up(pc::ProbΔ, lc::LogisticΔ, data::XData{E}) where{E <: eltype(F)} where{PC, LC, F}
    expFlowCircuit = ExpFlowΔ(pc, lc, num_examples(data), Float64);
    for n in expFlowCircuit
        exp_pass_up_node(n, data)
    end 
    expFlowCircuit
end

function exp_pass_up(fc::ExpFlowΔ, data::XData{E}) where{E <: eltype(F)} where{PC, LC, F}
    #TODO write resize_flows similar to flow_circuits
    #     and give as input the expFlowCircuit instead
    #expFlowCircuit = ExpFlowΔ(pc, lc, num_examples(data), Float64);
    for n in fc
        exp_pass_up_node(n, data)
    end
end

function exp_pass_up_node(node::ExpFlowNode{PC,LC,F}, data::XData{E}) where{E <: eltype(F)} where{PC, LC, F}
    pType = typeof(node.p_origin)
    fType = typeof(node.f_origin)

    if node.p_origin isa Prob⋁ && node.f_origin isa Logistic⋀Node
        #todo this ordering might be different than the ExpFlowNode children
        pthetas = [exp(node.p_origin.log_thetas[i])
                    for i in 1:length(node.p_origin.children) for j in 1:length(node.f_origin.children)]
        fthetas = [node.f_origin.thetas[j,:] # only taking the first class for now
            for i in 1:length(node.p_origin.children) for j in 1:length(node.f_origin.children)]

        node.f .= 0.0
        node.fg .= 0.0
        for z = 1:length(node.children)
            node.f  .+= pthetas[z] .* node.children[z].f
            node.fg .+= (pthetas[z] .* fthetas[z]) .* node.children[z].f
            node.fg .+= pthetas[z] .* node.children[z].fg
        end
    elseif node.p_origin isa Prob⋀ && node.f_origin isa Logistic⋀Node
        node.f .= node.children[1].f .* node.children[2].f # assume 2 children
        node.fg .= (node.children[1].f .* node.children[2].fg) .+
                   (node.children[2].f .* node.children[1].fg)

    elseif node.p_origin isa ProbLiteral 
        if node.f_origin isa Logistic⋀Node
            m = node.f_origin.children[1]
        elseif node.f_origin isa LogisticLiteral
            m = node.f_origin
        else
            error("Invalid Types of pairs {$pType} - {$fType}")
        end

        var = lit2var(literal(m))
        X = feature_matrix(data)
        if ispositive(node.p_origin) && ispositive(m)
            node.f[:, X[:, var] .!= 0 ] .= 1.0 # positive and missing observations
            node.f[:, X[:, var] .== 0 ] .= 0.0
        elseif isnegative(node.p_origin) && isnegative(m)
            node.f[:, X[:, var] .!= 1 ] .= 1.0 # negative and missing observations
            node.f[:, X[:, var] .== 1 ] .= 0.0
        else
            node.f .= 0.0
        end

        if node.f_origin isa Logistic⋀Node
            node.fg .= node.f .* transpose(node.f_origin.thetas)
        else
            node.fg .= 0.0
        end

    else
        error("Invalid Types of pairs {$pType} - {$fType}")
    end

end