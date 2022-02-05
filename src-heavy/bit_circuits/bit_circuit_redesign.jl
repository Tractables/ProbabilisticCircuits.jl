using CUDA, Random

struct BitsLiteral
    literal::Int
end

struct BitsInnerNode
    issum::Bool
end

const BitsNode = Union{BitsLiteral, BitsInnerNode}

struct SumEdge 
    parent_id::UInt32
    prime_id::UInt32
    sub_id::UInt32 # 0 means no sub
    logp::Float32
    tag::UInt8
end

struct MulEdge 
    parent_id::UInt32
    prime_id::UInt32
    sub_id::UInt32 # 0 means no sub
    tag::UInt8
end

const BitsEdge = Union{SumEdge,MulEdge}

hassub(x) = (x.sub_id != 0)

changetag(edge::SumEdge, tag) = 
    SumEdge(edge.parent_id, edge.prime_id, edge.sub_id, edge.logp, tag)

changetag(edge::MulEdge, tag) = 
    MulEdge(edge.parent_id, edge.prime_id, edge.sub_id, tag)

function tag_index(i,n)
    x = zero(UInt8)
    (i==1) && (x |= one(UInt8))
    (i==n) && (x |= (one(UInt8) << 1)) 
    x
end

@inline isfirst(x) = ((x & one(x)) != zero(x))
@inline islast(x) = ((x & one(x) << 1) != zero(x))
@inline ispartial(x) = ((x & one(x) << 2) != zero(x))
@inline isonlysubedge(x) = ((x & one(x) << 3) != zero(x))
    
rotate(edge::SumEdge) = 
    SumEdge(edge.parent_id, edge.sub_id, edge.prime_id, edge.logp, edge.tag)

rotate(edge::MulEdge) = 
    MulEdge(edge.parent_id, edge.sub_id, edge.prime_id, edge.tag)

const EdgeLayer = Vector{BitsEdge}

struct FlatVector{V <: AbstractVector}
    vectors::V
    ends::Vector{Int}
end

function FlatVector(vectors)
    FlatVector(vcat(vectors...), 
                cumsum(map(length, vectors)))
end

abstract type AbstractBitsProbCircuit end

struct BitsProbCircuit <: AbstractBitsProbCircuit
    nodes::Vector{BitsNode}
    edge_layers_up::FlatVector{EdgeLayer}
    edge_layers_down::FlatVector{EdgeLayer}
    down2upedge::Vector{Int32}
    BitsProbCircuit(n,e1,e2,d) = begin
        @assert length(e1.vectors) == length(e2.vectors)
        @assert allunique(e1.ends) "No empty layers allowed"
        @assert allunique(e2.ends) "No empty layers allowed"
        new(n,e1,e2,d)
    end
end

struct NodeInfo
    prime_id::Int
    prime_layer_id::Int
    sub_id::Int
    sub_layer_id::Int
end

@inline combined_layer_id(ni::NodeInfo) = max(ni.prime_layer_id, ni.sub_layer_id)

function BitsProbCircuit(pc; eager_materialize=true, 
                             collapse_elements=true)
    nodes = BitsNode[]
    node_layers = Vector{Int}[]
    outedges = Vector{Tuple{BitsEdge,Int,Int}}[]
    uplayers = EdgeLayer[]

    add_node(bitsnode, layer_id) = begin
        # add node globally
        push!(nodes, bitsnode)
        push!(outedges, Vector{Tuple{BitsEdge,Int,Int}}())
        id = length(nodes)
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
            push!(uplayers, EdgeLayer())
        end
        push!(uplayers[parent_layer_id], edge)

        # record out edges for downward pass
        id_within_uplayer = length(uplayers[parent_layer_id])
        upedgeinfo = (edge, parent_layer_id, id_within_uplayer)
        @assert uplayers[upedgeinfo[2]][upedgeinfo[3]] == edge "upedge $(uplayers[upedgeinfo[2]][upedgeinfo[3]]) does not equal $edge"

        push!(outedges[child_info.prime_id], upedgeinfo)
        if hassub(child_info)
            push!(outedges[child_info.sub_id], upedgeinfo)
        end
    end

    f_leaf(node) = begin
        node_id = add_node(BitsLiteral(literal(node)), 0)
        NodeInfo(node_id, 0, 0, 0)
    end
    
    f_inner(node, children_info) = begin
        if (length(children_info) == 1 
            && (!eager_materialize || !hassub(children_info[1])))
            # this is a pass-through node
            children_info[1]
        elseif (collapse_elements && ismul(node) && length(children_info) == 2 
                && children_info[1].sub_id == 0 && !hassub(children_info[2]))
            # this is a simple conjunctive element that we collapse into an edge
            prime_layer_id = children_info[1].prime_layer_id
            sub_layer_id = children_info[2].prime_layer_id
            NodeInfo(children_info[1].prime_id, prime_layer_id, 
                        children_info[2].prime_id, sub_layer_id)
        else
            layer_id = maximum(combined_layer_id, children_info) + 1
            node_id = add_node(BitsInnerNode(issum(node)), layer_id)
            if issum(node)
                for i = 1:length(children_info)
                    logp = node.log_probs[i]
                    child_info = children_info[i]
                    tag = tag_index(i, length(children_info))
                    edge = SumEdge(node_id, child_info.prime_id, child_info.sub_id, logp, tag)
                    add_edge(layer_id, edge, child_info)
                end
            else
                @assert ismul(node)
                single_infos = filter(x -> !hassub(x), children_info)
                double_infos = filter(x -> hassub(x), children_info)
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
                for i = 1:length(double_infos)
                    child_info = double_infos[i]
                    tag = tag_index(i, length(double_infos))
                    edge = MulEdge(node_id, child_info.prime_id, child_info.sub_id, tag)
                    add_edge(layer_id, edge, child_info)
                end
            end
            NodeInfo(node_id, layer_id, 0, 0)
        end
    end

    root_info = foldup_aggregate(pc, f_leaf, f_inner, NodeInfo)

    if hassub(root_info)
        # manually materialize root node
        @assert ismul(pc)
        @assert num_children(pc) == 1
        layer_id = root_info.layer_id + 1
        node_id = add_node(BitsInnerNode(false), layer_id)
        edge = MulEdge(node_id, root_info.prime_id, root_info.sub_id, tag_index(1,1))
        add_edge(layer_id, edge, root_info)
    end

    flatuplayers = FlatVector(uplayers)

    downedges = EdgeLayer()
    downlayerends = Int[]
    down2upedges = Int32[]
    uplayerstarts = [1, map(x -> x+1, flatuplayers.ends[1:end-1])...]
    @assert length(node_layers[end]) == 1 && isempty(outedges[node_layers[end][1]])
    for node_layer in node_layers[end-1:-1:1]
        for node_id in node_layer
            prime_edges = filter(e -> e[1].prime_id == node_id, outedges[node_id])
            partial = (length(prime_edges) != length(outedges[node_id]))
            for i in 1:length(prime_edges)
                edge = prime_edges[i][1]
                if edge.prime_id == node_id
                    # record the index in flatuplayers corresponding to this downedge
                    upedgeindex = uplayerstarts[prime_edges[i][2]] + prime_edges[i][3] - 1
                    push!(down2upedges, upedgeindex)

                    # update the tag and record down edge
                    tag = tag_index(i, length(prime_edges))
                    # tag whether this series of edges is partial or complete
                    partial && (tag |= (one(UInt8) << 2))
                    # tag whether this sub edge is the only outgoing edge from sub
                    if hassub(edge) && length(outedges[edge.sub_id]) == 1
                        tag |= (one(UInt8) << 3)
                    end
                    edge = changetag(edge, tag)
                    push!(downedges, edge)
                end
            end
        end
        # record new end of layer
        if (!isempty(downedges) && isempty(downlayerends)) || (length(downedges) > downlayerends[end])
            push!(downlayerends, length(downedges))
        end
    end
    
    BitsProbCircuit(nodes, flatuplayers, FlatVector(downedges, downlayerends), down2upedges)
end

const CuEdgeLayer = CuVector{BitsEdge}

struct CuBitsProbCircuit <: AbstractBitsProbCircuit
    
    nodes::CuVector{BitsNode}
    edge_layers_up::FlatVector{<:CuEdgeLayer}
    edge_layers_down::FlatVector{<:CuEdgeLayer}
    down2upedge::CuVector{Int32}

    CuBitsProbCircuit(bpc::BitsProbCircuit) = begin
        nodes = cu(bpc.nodes)
        edge_layers_up = FlatVector(cu(bpc.edge_layers_up.vectors), 
                            bpc.edge_layers_up.ends)
        edge_layers_down = FlatVector(cu(bpc.edge_layers_down.vectors), 
                                      bpc.edge_layers_down.ends)
        down2upedge = cu(bpc.down2upedge)
        new(nodes, edge_layers_up, edge_layers_down, down2upedge)
    end
end
