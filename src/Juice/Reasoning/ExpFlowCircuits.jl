########################
# Do not use for now
######################
#####################
# Expectation Flow circuits
# For use of algorithms depending on pairs of nodes of two circuits
#####################

"A expectation circuit node that has pair of origins of type PC and type LC"
abstract type DecoratorΔNodePair{PC<:ΔNode, LC<:ΔNode} <: ΔNode end

abstract type ExpFlowΔNode{PC, LC, F} <: DecoratorΔNodePair{PC, LC} end

struct UpExpFlow{PC, LC, F} <: ExpFlowΔNode{PC, LC, F}
    p_origin::PC
    f_origin::LC
    children::Vector{<:ExpFlowΔNode{<:PC, <:LC, <:F}}
    exp::F
end


"""
Construct a upward expectation flow circuit from a given pair of PC and LC circuits
Note that its assuming the two circuits share the same vtree
"""
function ExpFlowΔ(pc::ProbΔ, lc::LogisticΔ, batch_size::Int, ::Type{El}) where El
    pc_type = circuitnodetype(pc)
    lc_type = circuitnodetype(lc)

    F = (El == Bool) ? BitVector : Vector{El}
    fmem = () -> Vector{El}(undef, batch_size)  #some_vector(El, batch_size) # note: fmem's return type will determine type of all UpFlows in the circuit (should be El)

    root_pc = pc[end]
    root_lc = lc[end]
    
    cache = Dict{Pair{ΔNode, ΔNode}, ExpFlowΔNode}()
    sizehint!(cache, (length(pc) + length(lc))*4÷3)
    expFlowCircuit = Vector{ExpFlowΔNode}()

    function ExpflowTraverse(n::Prob⋀, m::Logistic⋁) 
        # PSDD root is AND, LC root is OR (after considering the bias)
        # TODO (pashak) temp solution to get nodes of same type
        #               ignoring the bias for now
        get!(cache, Pair(n, m)) do
            @assert length(m.children) == 1
            ExpflowTraverse(n, m.children[1])
        end
    end

    function ExpflowTraverse(n::Prob⋁, m::Logistic⋁) 
        get!(cache, Pair(n, m)) do
            children = [ ExpflowTraverse(i, j) for i in n.children for j in m.children]
            node = UpExpFlow{pc_type,lc_type, F}(n, m, children, fmem())
            push!(expFlowCircuit, node)
            return node
        end
    end
    function ExpflowTraverse(n::Prob⋀, m::Logistic⋀) 
        get!(cache, Pair(n, m)) do
            children = [ ExpflowTraverse(z[1], z[2]) for z in zip(n.children, m.children) ]
            node = UpExpFlow{pc_type,lc_type, F}(n, m, children, fmem())
            push!(expFlowCircuit, node)
            return node
        end
    end
    function ExpflowTraverse(n::ProbLiteral, m::Logistic⋁) 
        get!(cache, Pair(n, m)) do
            children = Vector{ExpFlowΔNode{pc_type,lc_type, F}}() # TODO
            node = UpExpFlow{pc_type,lc_type, F}(n, m, children, fmem())
            push!(expFlowCircuit, node)
            return node
        end
    end

    ExpflowTraverse(root_pc, root_lc)
    expFlowCircuit
end


function exp_pass_up(pc::ProbΔ, lc::LogisticΔ, data::XData{E}) where{E <: eltype(F)} where{PC, LC, F}
    #TODO write resize_flows similar to flow_circuits
    #     and give as input the expFlowCircuit instead
    expFlowCircuit = ExpFlowΔ(pc, lc, num_examples(data), Float64);
    for n in expFlowCircuit
        exp_pass_up_node(n, data)
    end 
    expFlowCircuit
end


## Not finished, do not use for now
function exp_pass_up_node(node::ExpFlowΔNode{PC,LC,F}, data::XData{E}) where{E <: eltype(F)} where{PC, LC, F}
    pType = typeof(node.p_origin)
    fType = typeof(node.f_origin)

    if  node.p_origin isa Prob⋁ && node.f_origin isa Logistic⋁
        params = [exp(node.p_origin.log_thetas[i]) #* node.f_origin.thetas[j]
                    for i in 1:length(node.p_origin.children) for j in 1:length(node.f_origin.children)]
        exps = [n.exp for n in node.children]
        node.exp .= log.(sum.(params .* exps))
    elseif node.p_origin isa Prob⋀ && node.f_origin isa Logistic⋀
        ll = [ n.exp for n in node.children ]
        # println(ll)
        # println(sum(ll, dims=1))
        # node.exp .= sum(ll, dims=1)[1,:]  # because we are in log domain
    elseif node.p_origin isa ProbLiteral && node.f_origin isa Logistic⋁
        println("Leaf")
    else
        error("Invalid Types of pairs {$pType} - {$fType}")
    end

end