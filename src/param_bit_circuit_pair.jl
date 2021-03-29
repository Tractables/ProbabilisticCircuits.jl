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
const ELEMENTS_LENGTH = 3

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

Elements are represented by a 3xE matrix, where 
  * elements[1,:] is the Product pair node id,
  * elements[2,:] is the (left,left) child node id 
  * elements[3,:] is the (right right) child node id 

  Elements belonging to node pair `i` are `elements[:, nodes[1,i]:nodes[2,i]]` 

Parents are represented as a one dimentional array  
    Parents belonging to node pair `i` are `parents[nodes[3,i]:nodes[4,i]]`

"""
struct BitCircuitPair{V,M}
    layers::Vector{V}
    nodes::M
    elements::M
    parents::V
end

struct ParamBitCircuitPair{V,M, WPC, WLC}
    pc_bit::BitCircuit{V,M}
    lc_bit::BitCircuit{V,M}
    pair_bit::BitCircuitPair{V,M}
    pc_params::WPC
    lc_params::WLC
end

function ParamBitCircuitPair(pc::ProbCircuit, lc::LogisticCircuit; Float=Float32)
    pc_thetas::Vector{Float} = Vector{Float}()
    lc_thetas::Vector{Vector{Float}} = Vector{Vector{Float}}()

    sizehint!(pc_thetas, num_edges(pc))

    pc_cache = Dict{Node, NodeId}() # only for sum nodes
    lc_cache = Dict{Node, NodeId}() # only for sum nodes

    lc_num_classes = num_classes(lc);
    pc_on_decision(n, cs, layer_id, decision_id, first_element, last_element) = begin
        if isnothing(n) # this decision node is not part of the PC
            push!(pc_thetas, zero(Float))
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

    pc_bit = BitCircuit(pc, num_variables(pc); on_decision=pc_on_decision)
    lc_bit = BitCircuit(lc, num_variables(pc); on_decision=lc_on_decision)
    bcp = BitCircuitPair(pc, lc; on_sum_callback = pbc_callback, pc_cache, lc_cache)
    
    lc_thetas_reshaped = permutedims(hcat(lc_thetas...), (2, 1))
    ParamBitCircuitPair(pc_bit, lc_bit, bcp, pc_thetas, lc_thetas_reshaped)
end


function BitCircuitPair(pc::ProbCircuit, lc::LogisticCircuit; 
    on_sum_callback=noop, 
    pc_cache=nothing, lc_cache=nothing, property_check=false)

    if property_check
        @assert num_variables(pc) == num_variables(lc)
        vtree1::Vtree = infer_vtree(pc);
        @assert respects_vtree(lc, vtree1);
    end

    num_features = num_variables(pc)
    num_leafs = 4*num_features
    layers::Vector{Vector{NodePairId}} = Vector{NodePairId}[collect(1:num_leafs)]
    nodes::Vector{NodePairId} = zeros(NodePairId, NODES_LENGTH*num_leafs)
    elements::Vector{NodePairId} = NodePairId[]
    parents::Vector{Vector{NodePairId}} = Vector{NodePairId}[NodePairId[] for i = 1:num_leafs]
    last_dec_id::NodePairId = num_leafs
    last_el_id::NodePairId = zero(NodePairId)

    cache = Dict{Pair{Node, Node}, NodePairIds}()

    func(n,m) = begin
        throw("Unsupported pair of nodes!! $n, $m")
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
                    push!(results[i], func(children(n)[i], children(m)[j]))
                end
            end

            # Do not move this, otherwise el_ids would be wrong.
            first_el_id::NodePairId = last_el_id + one(NodePairId)
            layer_id::NodePairId = zero(NodePairId)
            push!(parents, NodePairId[])

            # Add processed children to bitcircuitpair
            for i in 1:num_children(n)
                for j in 1:num_children(m)
                    cur = results[i][j];  
                    layer_id = max(layer_id, cur.layer_id)
                    last_el_id += one(NodePairId)
                    
                    if typeof(cur) == ProdNodePairIds
                        push!(elements, last_dec_id, cur.left_left_id, cur.right_right_id);
                        @inbounds push!(parents[cur.left_left_id], last_el_id)
                        @inbounds push!(parents[cur.right_right_id], last_el_id)
                    else
                        !property_check &&  @assert children(m)[1] isa LogisticLeafNode
                        push!(elements, last_dec_id, cur.node_id, cur.node_id);
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
    func(pc, lc)

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
            !property_check && @assert i <= num_leafs "Only root and leaf nodes can have no parents: $i <= $num_leafs"
        end
    end
    return BitCircuitPair(layers, nodes_m, elements_m, parents_m)
end


######################
## Helper Functions ##
######################

import LogicCircuits: num_nodes, num_elements, num_features, num_leafs, nodes, elements
import LogicCircuits: to_gpu, to_cpu, isgpu #extend

# most of this are identical with bitcircuit maybe make the BitCircuitPair a subtype of BitCircuit?
nodes(c::BitCircuitPair) = c.nodes
elements(c::BitCircuitPair) = c.elements

num_nodes(c::BitCircuitPair) = size(c.nodes, 2)

num_elements(c::BitCircuitPair) = size(c.elements, 2)

to_gpu(c::BitCircuitPair) = 
    BitCircuitPair(map(to_gpu, c.layers), to_gpu(c.nodes), to_gpu(c.elements), to_gpu(c.parents))

to_cpu(c::BitCircuitPair) = 
    BitCircuitPair(map(to_cpu, c.layers), to_cpu(c.nodes), to_cpu(c.elements), to_cpu(c.parents))

isgpu(c::BitCircuitPair{<:CuArray,<:CuArray}) = true
isgpu(c::BitCircuitPair{<:Array,<:Array}) = false


#############################
## Param Helper functions ###
#############################

pc_params(c::ParamBitCircuitPair) = c.pc_params
lc_params(c::ParamBitCircuitPair) = c.lc_params

num_nodes(c::ParamBitCircuitPair) = num_nodes(c.pair_bit)
num_elements(c::ParamBitCircuitPair) = num_elements(c.pair_bit)
num_features(c::ParamBitCircuitPair) = num_features(c.pair_bit)
num_leafs(c::ParamBitCircuitPair) = num_leafs(c.pair_bit)

nodes(c::ParamBitCircuitPair) = nodes(c.pair_bit)
elements(c::ParamBitCircuitPair) = elements(c.pair_bit)


to_gpu(c::ParamBitCircuitPair) = 
    ParamBitCircuitPair(to_gpu(c.pc_bit), to_gpu(c.lc_bit),
        to_gpu(c.pair_bit), to_gpu(c.pc_params), to_gpu(c.lc_params))

to_cpu(c::ParamBitCircuitPair) = 
    ParamBitCircuitPair(to_cpu(c.pc_bit), to_cpu(c.lc_bit), to_cpu(c.pair_bit), to_cpu(c.pc_params), to_cpu(c.lc_params))

isgpu(c::ParamBitCircuitPair) = 
    isgpu(c.pc_bit) && isgpu(c.lc_bit) && isgpu(c.pair_bit) && isgpu(c.pc_params) && isgpu(c.lc_params)
