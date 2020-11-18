using CUDA

export NodePairIds, 
       BitCircuitPair,
       SumNodePairIds,
       ProdNodePairIds,
       BitCircuitPair,
       ParamBitCircuitPair

const NodeId = Int32 

"Integer identifier for a circuit pair node"
const NodePairId = Int32 

"The BitCircuitPair ids associated with a node"
abstract type NodePairIds end

mutable struct SumNodePairIds <: NodePairIds 
    layer_id::NodePairId
    node_id::NodePairId
end

mutable struct ProdNodePairIds <: NodePairIds
    layer_id::NodePairId
    # Assuming both Product nodes have two children
    left_left_id::NodePairId
    right_right_id::NodePairId

    ProdNodePairIds(ll::SumNodePairIds, rr::SumNodePairIds) = begin
        l = max(ll.layer_id, rr.layer_id)
        new(l, ll.node_id, rr.node_id)
    end 
end


const NODES_LENGTH = 6
const ELEMENTS_LENGTH = 5

"""
A bit circuit pair is a low-level representation of pairs of nodes traversed for a pair of probabilistic circuit logical circuit structure.
For example, this is used for taking expectation of a Logistic/Regression circuit w.r.t. a probabilistic circuit.

The wiring of the circuit is captured by two matrices: nodes and elements.
  * Nodes are either leafs (input) or pair of Sum nodes
  * Elements are Pair of Product nodes 
  * In addition, there is a vector of layers, where each layer is a list of node ids.
    Layer 1 is the leaf/input layer. Layer end is the circuit root.
  * And there is a vector of parents, pointing to element id parents of decision nodes.

Nodes are represented as a 6xN matrix where
  * nodes[1,:] is the first child pair id 
  * nodes[2,:] is the last child pair id 
  * nodes[3,:] is the first parent index 
  * nodes[4,:] is the last parent index 
  * nodes[5,:] is the id of corresponding node from the PC (first circuit)
  * nodes[6,:] is the id of corresponding node from the LC (first circuit)


  Elements belonging to node pair `i` are `elements[:, nodes[1,i]:nodes[2,i]]` 
  Parents belonging to node pair `i` are `parents[nodes[3,i]:nodes[4,i]]`

Elements are represented by a 5xE matrix, where 
  * elements[1,:] is the Product pair node id,
  * elements[2,:] is the (left,left) child node id 
  * elements[3,:] is the (right right) child node id 
  * elements[4,:] i for debugging, remove later (can assume the child ordering)
  * elements[5,:] j
"""
struct BitCircuitPair{V,M}#, WPC, WLC}
    layers::Vector{V}
    nodes::M
    elements::M
    parents::V
end


struct ParamBitCircuitPair{V,M, WPC, WLC}
    pc_bit::BitCircuit{V,M}
    lc_bit::BitCircuit{V,M}
    bcp::BitCircuitPair{V,M}
    pc_params::WPC
    lc_params::WLC
end

function ParamBitCircuitPair(pc::ProbCircuit, lc::LogisticCircuit, data; reset=true)
    pc_thetas::Vector{Float64} = Vector{Float64}()
    lc_thetas::Vector{Vector{Float32}} = Vector{Vector{Float32}}()

    sizehint!(pc_thetas, num_edges(pc))

    pc_cache = Dict{Node, NodeId}() # only for sum nodes
    lc_cache = Dict{Node, NodeId}() # only for sum nodes

    lc_num_classes = num_classes(lc);

    pc_on_decision(n, cs, layer_id, decision_id, first_element, last_element) = begin
        if isnothing(n) # this decision node is not part of the PC
            push!(pc_thetas, 0.0)
        else
            pc_cache[n] = decision_id
            append!(pc_thetas, n.log_probs)
        end
    end

    lc_on_decision(m, cs, layer_id, decision_id, first_element, last_element) = begin
        if isnothing(m)
            throw("here, some node is not part of the logistic circuit")
            # push!(lc_thetas, zeros(Float32, lc_num_classes))            
        else
            lc_cache[m] = decision_id
            for theta in eachrow(m.thetas)
                push!(lc_thetas, theta)
            end
        end
    end

    pbc_callback(n, m, results, layer_id, last_dec_id, first_el_id, last_el_id) = begin
        nothing
    end

    pc_bit = BitCircuit(pc, data; reset=reset, on_decision=pc_on_decision)
    lc_bit = BitCircuit(lc, data; reset=reset, on_decision=lc_on_decision)
    bcp = BitCircuitPair(pc, lc; reset, on_sum_callback = pbc_callback, pc_cache, lc_cache)
    
    lc_thetas_reshaped = permutedims(hcat(lc_thetas...), (2, 1))

    ParamBitCircuitPair(pc_bit, lc_bit, bcp, pc_thetas, lc_thetas_reshaped)
end


function BitCircuitPair(pc::ProbCircuit, lc::LogisticCircuit; reset=true, on_sum_callback=noop, 
    pc_cache=nothing, lc_cache=nothing)

    @assert num_variables(pc) == num_variables(lc)
    # TODO check if they both have same vtree

    num_features = num_variables(pc)
    num_leafs = 4*num_features
    layers::Vector{Vector{NodePairId}} = Vector{NodePairId}[collect(1:num_leafs)]
    nodes::Vector{NodePairId} = zeros(NodePairId, NODES_LENGTH*num_leafs)
    elements::Vector{NodePairId} = NodePairId[]
    parents::Vector{Vector{NodePairId}} = Vector{NodePairId}[NodePairId[] for i = 1:num_leafs]
    last_dec_id::NodePairId = 4*num_features
    last_el_id::NodePairId = zero(NodePairId)

    cache = Dict{Pair{Node, Node}, NodePairIds}()

    func(n,m) = begin
        throw("This should not happen!! $n, $m")
    end

    function func(n::Union{PlainProbLiteralNode, StructProbLiteralNode}, 
                    m::LogisticLiteralNode)::NodePairIds
        get!(cache, Pair(n, m)) do
            @assert variable(n) == variable(m)
            if ispositive(n) && ispositive(m)
                SumNodePairIds(one(NodePairId), NodePairId(variable(n)))
            elseif !ispositive(n) && ispositive(m)
                SumNodePairIds(one(NodePairId), NodePairId(variable(n) + num_features))
            elseif ispositive(n) && !ispositive(m)
                SumNodePairIds(one(NodePairId), NodePairId(variable(n) + 2*num_features))
            elseif !ispositive(n) && !ispositive(m)
                SumNodePairIds(one(NodePairId), NodePairId(variable(n) + 3*num_features))
            end
        end
    end

    function func(n::Union{PlainProbLiteralNode, StructProbLiteralNode}, 
                    m::Logistic⋁Node)::NodePairIds
        
        @assert num_children(m) == 1
        func(n, children(m)[1])
    end

    function func(n::Union{PlainMulNode, StructMulNode}, 
            m::Logistic⋀Node)::NodePairIds
        
        get!(cache, Pair(n, m)) do
            @assert num_children(n) == 2
            @assert num_children(m) == 2

            ll = func(children(n)[1], children(m)[1])
            rr = func(children(n)[2], children(m)[2])
            ProdNodePairIds(ll, rr)
        end
    end


    function func(n::Union{PlainSumNode, StructSumNode}, 
                    m::Logistic⋁Node)::NodePairIds
        
        get!(cache, Pair(n, m)) do 
            
        
            # First process the children
            results::Vector{Vector{NodePairIds}} = NodePairIds[]
            for i in 1:num_children(n)
                push!(results, NodePairIds[])
                for j in 1:num_children(m)
                    #results[i][j] = func(children(n)[i], children(m)[j])
                    push!(results[i], func(children(n)[i], children(m)[j]))
                end
            end

            first_el_id::NodePairId = last_el_id + one(NodePairId)
            layer_id::NodePairId = zero(NodePairId)
            push!(parents, NodePairId[])

            

            for i in 1:num_children(n)
                for j in 1:num_children(m)
                    cur = results[i][j];  
                    layer_id = max(layer_id, cur.layer_id)

                    # if !(children(m)[1] isa LogisticLeafNode)
                    last_el_id += one(NodePairId)
                    # end


                    if typeof(cur) == ProdNodePairIds
                        push!(elements, last_dec_id, cur.left_left_id, cur.right_right_id, NodeId(i), NodeId(j));
                        @inbounds push!(parents[cur.left_left_id], last_el_id)
                        @inbounds push!(parents[cur.right_right_id], last_el_id)
                    else
                        @assert children(m)[1] isa LogisticLeafNode
                        push!(elements, last_dec_id, cur.node_id, cur.node_id, NodeId(i), NodeId(j));
                        @inbounds push!(parents[cur.node_id], last_el_id)
                        @inbounds push!(parents[cur.node_id], last_el_id)
                    end
                end
            end
            layer_id += one(NodePairId)
            length(layers) < layer_id && push!(layers, NodePairId[])

            last_dec_id::NodePairId += one(NodePairId)

            PC_ID = pc_cache !== nothing ? pc_cache[n] : zero(NodePairId)
            LC_ID = lc_cache !== nothing ? lc_cache[m] : zero(NodePairId)

            push!(nodes, first_el_id, last_el_id, zero(NodePairId), zero(NodePairId), PC_ID, LC_ID)
            push!(layers[layer_id], last_dec_id)
            on_sum_callback(n, m, results, layer_id, last_dec_id, first_el_id, last_el_id)
            SumNodePairIds(layer_id, last_dec_id)
        end
    end

    # Call on roots
    func(pc, children(lc)[1]) # skipping bias node of LC

    nodes_m = reshape(nodes, NODES_LENGTH, :)
    elements_m = reshape(elements, ELEMENTS_LENGTH, :)
    parents_m = Vector{NodePairId}(undef, size(elements_m,2)*2)

    last_parent = zero(NodePairId)[]
    @assert last_dec_id == size(nodes_m,2) == size(parents,1)
    @assert sum(length, parents) == length(parents_m)
    for i in 1:last_dec_id-1
        if !isempty(parents[i])
            nodes_m[3,i] = last_parent + one(NodePairId)
            parents_m[last_parent + one(NodePairId):last_parent + length(parents[i])] .= parents[i] 
            last_parent += length(parents[i])
            nodes_m[4,i] = last_parent
        else
            @assert i <= num_leafs "Only root and leaf nodes can have no parents: $i <= $num_leafs"
        end
    end
    return BitCircuitPair(layers, nodes_m, elements_m, parents_m)
end




