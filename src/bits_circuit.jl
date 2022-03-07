using CUDA

export BitsProbCircuit, CuBitsProbCircuit, update_parameters

###############################################
# bits representation of nodes
###############################################

abstract type BitsNode end

struct BitsInput{D <: InputDist} <: BitsNode
    variable::Var # for now assume single variable
    dist::D
end

dist(n::BitsNode) = nothing 
dist(n::BitsInput) = n.dist

function bits(in::PlainInputNode, heap)
    vars = Var.(randvars(in))
    bits_dist = bits(dist(in), heap)
    BitsInput(vars..., bits_dist)
end

update_dist(pcnode, bitsnode::BitsInput, heap) =
    pcnode.dist = unbits(bitsnode.dist, heap)

abstract type BitsInnerNode <: BitsNode end

struct BitsSum <: BitsInnerNode end
struct BitsMul <: BitsInnerNode end

###############################################
# bits representation of edges
###############################################

abstract type AbstractBitsEdge end

struct SumEdge <: AbstractBitsEdge
    parent_id::UInt32
    prime_id::UInt32
    sub_id::UInt32 # 0 means no sub
    logp::Float32
    tag::UInt8
end

struct MulEdge  <: AbstractBitsEdge
    parent_id::UInt32
    prime_id::UInt32
    sub_id::UInt32 # 0 means no sub
    tag::UInt8
end

const BitsEdge = Union{SumEdge,MulEdge}

hassub(x) = !iszero(x.sub_id)

rotate(edge::SumEdge) = 
    SumEdge(edge.parent_id, edge.sub_id, edge.prime_id, edge.logp, edge.tag)

rotate(edge::MulEdge) = 
    MulEdge(edge.parent_id, edge.sub_id, edge.prime_id, edge.tag)

# tags

@inline tag_at(tag, i) = (tag | (one(UInt8) << i)) 
@inline tagged_at(tag, i) = ((tag & one(tag) << i) != zero(tag))

@inline isfirst(tag) = tagged_at(tag, 0)
@inline islast(tag) = tagged_at(tag, 1)
"whether this series of edges is partial or complete"
@inline ispartial(tag) = tagged_at(tag, 2)
"whether this sub edge is the only outgoing edge from sub"
@inline isonlysubedge(tag) = tagged_at(tag, 3)

function tag_firstlast(i,n)
    tag = zero(UInt8)
    (i==1) && (tag = tag_at(tag, 0))
    (i==n) && (tag = tag_at(tag, 1)) 
    tag
end

tagpartial(tag) = tag_at(tag, 2)
tagonlysubedge(tag) = tag_at(tag, 3)

changetag(edge::SumEdge, tag) = 
    SumEdge(edge.parent_id, edge.prime_id, edge.sub_id, edge.logp, tag)

changetag(edge::MulEdge, tag) = 
    MulEdge(edge.parent_id, edge.prime_id, edge.sub_id, tag)
    
###############################################
# bits representation of nested vectors
###############################################

"An `isbits` representation of a `AbstractVector{<:AbstractVector}`"
struct FlatVectors{V <: AbstractVector} <: AbstractVector{V}
    vectors::V
    ends::Vector{Int}
end

function FlatVectors(vectors::AbstractVector{<:AbstractVector})
    flatvectors = vcat(vectors...)
    ends = cumsum(map(length, vectors))
    FlatVectors(flatvectors, ends)
end

layer_end(fv, i) =
    fv.ends[i]

layer_start(fv, i) =
    (i == 1) ? 1 : layer_end(fv, i-1) + 1

import CUDA: cu #extend

cu(fv::FlatVectors) =
    FlatVectors(cu(fv.vectors), fv.ends)

import Base: size, getindex #extend

size(fv::FlatVectors) = size(fv.vectors)

getindex(fv::FlatVectors, idx) = getindex(fv.vectors, idx)

###############################################
# bits representation of circuit
###############################################

abstract type AbstractBitsProbCircuit end 
struct BitsProbCircuit <: AbstractBitsProbCircuit
    
    # all the nodes in the circuit
    nodes::Vector{BitsNode}

    # mapping from BitPC to PC nodes 
    nodes_map::Vector{ProbCircuit}

    # the ids of the subset of nodes that are inputs
    input_node_ids::Vector{UInt32}
    
    # layers of edges for upward pass
    edge_layers_up::FlatVectors{Vector{BitsEdge}}
    
    # layers of edges for downward pass
    edge_layers_down::FlatVectors{Vector{BitsEdge}}
    
    # mapping from downward pass edge id to upward pass edge id
    down2upedge::Vector{Int32}
    
    # memory used by input nodes for their parameters and parameter learning
    heap::Vector{Float32}

    # index of beginning and ending edge for the node in the BitEdges flatvector
    node_begin_end::Vector{Pair{UInt32, UInt32}}

    BitsProbCircuit(n, nm, in, e1, e2, d, heap, node_be) = begin
        @assert length(n) == length(nm) >= length(in) > 0
        @assert length(e1.vectors) == length(e2.vectors)
        @assert allunique(e1.ends) "No empty layers allowed"
        @assert allunique(e2.ends) "No empty layers allowed"
        new(n, nm, in, e1, e2, d, heap, node_be)
    end
end

struct CuBitsProbCircuit{BitsNodes <: BitsNode} <: AbstractBitsProbCircuit
    
    # all the nodes in the circuit
    nodes::CuVector{BitsNodes}

    # mapping from BitPC to PC nodes 
    nodes_map::Vector{ProbCircuit}

    # the ids of the subset of nodes that are inputs
    input_node_ids::CuVector{UInt32}
    
    # layers of edges for upward pass
    edge_layers_up::FlatVectors{<:CuVector{BitsEdge}}
    
    # layers of edges for downward pass
    edge_layers_down::FlatVectors{<:CuVector{BitsEdge}}
    
    # mapping from downward pass edge id to upward pass edge id
    down2upedge::CuVector{Int32}
    
    # memory used by input nodes for their parameters and parameter learning
    heap::CuVector{Float32}

    node_begin_end::CuVector{Pair{UInt32, UInt32}}

    CuBitsProbCircuit(bpc) = begin
        # find union of bits node types actually used in the circuit
        BitsNodes = mapreduce(typeof, (x,y) -> Union{x,y}, bpc.nodes)
        @assert Base.isbitsunion(BitsNodes)
        nodes = CuVector{BitsNodes}(bpc.nodes)
        input_node_ids = cu(bpc.input_node_ids)
        edge_layers_up = cu(bpc.edge_layers_up)
        edge_layers_down = cu(bpc.edge_layers_down)
        down2upedge = cu(bpc.down2upedge)
        heap = cu(bpc.heap)
        node_be = cu(bpc.node_begin_end)
        new{BitsNodes}(nodes, bpc.nodes_map, input_node_ids, 
                       edge_layers_up, edge_layers_down, down2upedge, heap, node_be)
    end
end

CuBitsProbCircuit(pc::ProbCircuit) =
    CuBitsProbCircuit(BitsProbCircuit(pc))

cu(bpc::BitsProbCircuit) = CuBitsProbCircuit(bpc)

###############################################
# converting a PC into a BitsPC
###############################################

struct NodeInfo
    prime_id::Int
    prime_layer_id::Int
    sub_id::Int # 0 means no sub
    sub_layer_id::Int
end

struct OutputInfo
    edge::BitsEdge
    parent_layer_id::Int
    id_within_uplayer::Int
end

function BitsProbCircuit(pc::ProbCircuit; eager_materialize=true, collapse_elements=true)

    nodes = BitsNode[]
    nodes_map = ProbCircuit[]
    input_node_ids = UInt32[]
    node_layers = Vector{Int}[]
    outputs = Vector{OutputInfo}[]
    uplayers = Vector{BitsEdge}[]
    heap = Float32[]

    add_node(pcnode, bitsnode, layer_id) = begin
        # add node globally
        push!(nodes, bitsnode)
        push!(nodes_map, pcnode)
        push!(outputs, OutputInfo[])
        id = length(nodes)
        # add index for input nodes
        if bitsnode isa BitsInput
            push!(input_node_ids, id)
        end
        # add node to node layers 
        while length(node_layers) <= layer_id
            push!(node_layers, Int[])
        end
        push!(node_layers[layer_id+1], id)
        id
    end

    add_edge(parent_layer_id, edge, child_info) = begin

        # introduce invariant that primes are never at a lower layer than subs 
        if hassub(child_info) && child_info.prime_layer_id < child_info.sub_layer_id
            edge = rotate(edge)
        end

        # record up edges for upward pass
        while length(uplayers) < parent_layer_id
            push!(uplayers, BitsEdge[])
        end
        push!(uplayers[parent_layer_id], edge)

        # record out edges for downward pass
        id_within_uplayer = length(uplayers[parent_layer_id])
        outputinfo = OutputInfo(edge, parent_layer_id, id_within_uplayer)
        @assert uplayers[outputinfo.parent_layer_id][outputinfo.id_within_uplayer] == edge

        push!(outputs[edge.prime_id], outputinfo)
        if hassub(edge)
            push!(outputs[edge.sub_id], outputinfo)
        end
    end

    f_input(node) = begin
        node_id = add_node(node, bits(node, heap), 0)
        NodeInfo(node_id, 0, 0, 0)
    end
    
    f_inner(node, children_info) = begin
        if (length(children_info) == 1 && (node !== pc)
            && (!eager_materialize || !hassub(children_info[1]))) 
            # this is a pass-through node
            children_info[1]
        elseif (collapse_elements && ismul(node) && length(children_info) == 2 
                && !hassub(children_info[1]) && !hassub(children_info[2]) && (node !== pc))
            # this is a simple conjunctive element that we collapse into an edge
            prime_layer_id = children_info[1].prime_layer_id
            sub_layer_id = children_info[2].prime_layer_id
            NodeInfo(children_info[1].prime_id, prime_layer_id, 
                     children_info[2].prime_id, sub_layer_id)
        else
            layer_id = 1 + maximum(children_info) do info
                max(info.prime_layer_id, info.sub_layer_id)
            end
            if issum(node)
                node_id = add_node(node, BitsSum(), layer_id)
                for i = 1:length(children_info)
                    param = params(node)[i]
                    child_info = children_info[i]
                    tag = tag_firstlast(i, length(children_info))
                    edge = SumEdge(node_id, child_info.prime_id, child_info.sub_id, param, tag)
                    add_edge(layer_id, edge, child_info)
                end
            else
                @assert ismul(node)
                node_id = add_node(node, BitsMul(), layer_id)
                # try to merge inputs without a sub into "double" edges
                children_info = merge_mul_inputs(children_info)
                for i = 1:length(children_info)
                    child_info = children_info[i]
                    tag = tag_firstlast(i, length(children_info))
                    edge = MulEdge(node_id, child_info.prime_id, child_info.sub_id, tag)
                    add_edge(layer_id, edge, child_info)
                end
            end
            NodeInfo(node_id, layer_id, 0, 0)
        end
    end

    root_info = foldup_aggregate(pc, f_input, f_inner, NodeInfo)
    @assert !hassub(root_info)

    flatuplayers = FlatVectors(uplayers)
    flatdownlayers, down2upedges = down_layers(node_layers, outputs, flatuplayers)

    node_begin_end = [Pair(typemax(UInt32), typemin(UInt32)) for i=1:length(nodes)]
    for i = 1:length(flatuplayers.vectors)
        pi = flatuplayers.vectors[i].parent_id
        l, r = node_begin_end[pi]
        node_begin_end[pi] = Pair( min(l, i), max(r, i) )
    end

    BitsProbCircuit(nodes, nodes_map, input_node_ids, 
                    flatuplayers, flatdownlayers, down2upedges, heap, node_begin_end)
end

function merge_mul_inputs(children_info)
    single_infos = filter(!hassub, children_info)
    double_infos = filter( hassub, children_info)
    for i = 1:2:length(single_infos)
        if i < length(single_infos)
            prime_layer_id = single_infos[i].prime_layer_id
            sub_layer_id = single_infos[i+1].prime_layer_id
            merged_info = NodeInfo(single_infos[i].prime_id, prime_layer_id, 
                                    single_infos[i+1].prime_id, sub_layer_id)
            single_infos[i] = merged_info
        end
        push!(double_infos, single_infos[i])
    end
    @assert length(double_infos) == length(children_info) - (length(single_infos) ÷ 2)
    double_infos
end

function down_layers(node_layers, outputs, flatuplayers)
    downedges = BitsEdge[]
    downlayerends = Int[]
    down2upedges = Int32[]
    @assert length(node_layers[end]) == 1 && isempty(outputs[node_layers[end][1]])
    for node_layer in node_layers[end-1:-1:1]
        for node_id in node_layer
            prime_outputs = filter(e -> e.edge.prime_id == node_id, outputs[node_id])
            partial = (length(prime_outputs) != length(outputs[node_id]))
            for i in 1:length(prime_outputs)
                prime_output = prime_outputs[i]
                edge = prime_output.edge
                @assert edge.prime_id == node_id
                # record the index in flatuplayers corresponding to this downedge
                upedgeindex = layer_start(flatuplayers, prime_output.parent_layer_id) + 
                                prime_output.id_within_uplayer - 1
                push!(down2upedges, upedgeindex)

                # update the tag and record down edge
                tag = tag_firstlast(i, length(prime_outputs))
                partial && (tag = tagpartial(tag))
                if hassub(edge) && length(outputs[edge.sub_id]) == 1
                    tag = tagonlysubedge(tag)
                end
                edge = changetag(edge, tag)
                push!(downedges, edge)
            end
        end
        # record new end of layer
        if (!isempty(downedges) && isempty(downlayerends)) || (length(downedges) > downlayerends[end])
            push!(downlayerends, length(downedges))
        end
    end
    flatdownlayers = FlatVectors(downedges, downlayerends)
    flatdownlayers, down2upedges
end

#####################
# retrieve parameters from BitsPC
#####################

"map parameters from BitsPC back to the ProbCircuit it was created from"
function update_parameters(bpc::AbstractBitsProbCircuit)
    nodemap = bpc.nodes_map
    
    # copy parameters from sum nodes
    edges = Vector(bpc.edge_layers_up.vectors)
    i = 1
    while i <= length(edges)
        @assert isfirst(edges[i].tag)
        parent = nodemap[edges[i].parent_id]
        if issum(parent)
            ni = num_inputs(parent)
            params(parent) .= map(e -> e.logp, edges[i:i+ni-1])
        else # parent is a product node
            ni = 1
            while !isfirst(edges[i+ni].tag)
                ni += 1
            end
        end
        i += ni
    end
    
    # copy parameters from input nodes
    nodes = Vector(bpc.nodes)
    input_ids = Vector(bpc.input_node_ids)
    heap = Vector(bpc.heap)
    for i in input_ids
        update_dist(nodemap[i], nodes[i], heap)
    end
    nothing
end