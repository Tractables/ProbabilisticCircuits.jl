using CUDA

export BitsProbCircuit, CuBitsProbCircuit, copy_parameters!


abstract type BitsNode end

struct BitsInputNode{D} <: BitsNode
    variable::UInt32
    dist::D
end

function bits(in::PlainInputNode, heap)
    vars = UInt32.(randvars(in))
    bits_dist = bits(dist(in), heap)
    BitsInputNode(vars..., bits_dist)
end

struct BitsInnerNode <: BitsNode
    issum::Bool
end

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
function unflatten(fv::FlatVector{V}) where V <: AbstractVector
    v = Vector{V}()
    startpoint = 1
    for endpoint in fv.ends
        push!(v, fv.vectors[startpoint:endpoint])
        startpoint = endpoint + 1
    end
    v
end

abstract type AbstractBitsProbCircuit end

struct BitsProbCircuit <: AbstractBitsProbCircuit
    nodes::Vector{BitsNode}
    input_node_idxs::Vector{UInt32}
    edge_layers_up::FlatVector{EdgeLayer}
    edge_layers_down::FlatVector{EdgeLayer}
    down2upedge::Vector{Int32}
    input_node_vars::Matrix{UInt32}
    input_node_params::Matrix{Float32}
    inparams_aggr_size::Int32
    sumnode2edges::Dict{PlainProbCircuit,Tuple{Int,Int,Int}}
    BitsProbCircuit(n, in, e1, e2, d, iv, ip, is, s2e) = begin
        @assert length(in) > 0
        @assert length(e1.vectors) == length(e2.vectors)
        @assert allunique(e1.ends) "No empty layers allowed"
        @assert allunique(e2.ends) "No empty layers allowed"
        new(n, in, e1, e2, d, iv, ip, is, s2e)
    end
end

struct NodeInfo
    prime_id::Int
    prime_layer_id::Int
    sub_id::Int
    sub_layer_id::Int
end

@inline combined_layer_id(ni::NodeInfo) = max(ni.prime_layer_id, ni.sub_layer_id)

function BitsProbCircuit(pc; eager_materialize=true, collapse_elements=true)

    assign_input_node_ids!(pc) # assign every input node an unique id

    nodes = BitsNode[]
    input_node_idxs = Vector{UInt32}()
    node_layers = Vector{Int}[]
    outedges = Vector{Tuple{BitsEdge,Int,Int}}[]
    uplayers = EdgeLayer[]
    
    sumnode2edges = Dict{PlainProbCircuit,Tuple{Int,Int,Int}}()

    add_node(bitsnode, layer_id) = begin
        # add node globally
        push!(nodes, bitsnode)
        push!(outedges, Vector{Tuple{BitsEdge,Int,Int}}())
        id = length(nodes)
        # add index for input nodes
        if bitsnode isa BitsInputNode
            push!(input_node_idxs, convert(UInt32, id))
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
        node_id = add_node(
            BitsInputNode(global_id(node), dist_type_id(dist(node)), num_parameters_node(node, true)), 
            0
        )
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
                    logp = node.params[i]
                    child_info = children_info[i]
                    tag = tag_index(i, length(children_info))
                    edge = SumEdge(node_id, child_info.prime_id, child_info.sub_id, logp, tag)
                    add_edge(layer_id, edge, child_info)
                end
                edge_end_id = length(uplayers[layer_id])
                edge_start_id = edge_end_id - length(children_info) + 1
                sumnode2edges[node] = (layer_id, edge_start_id, edge_end_id)
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

    ninput_nodes = num_input_nodes(pc)
    max_nvars = max_nvars_per_input(pc)
    max_nparams = max_nparams_per_input(pc)

    input_node_vars = zeros(UInt32, ninput_nodes, max_nvars)
    input_node_params = zeros(Float32, ninput_nodes, max_nparams)
    
    foreach(pc) do n
        if n isa PlainInputNode
            idx = global_id(n)
            nvars = num_randvars(n)
            @inbounds @views input_node_vars[idx, 1:nvars] .= collect(randvars(n))
            d = dist(n)
            if d isa Indicator
                @inbounds input_node_params[idx, 1] = ifelse(d.sign, one(Float32), zero(Float32))
            elseif d isa BernoulliDist
                @inbounds input_node_params[idx, 1] = d.logp
            elseif d isa CategoricalDist
                ncats = length(d.logps)
                @inbounds @views input_node_params[idx, 1:ncats] .= d.logps
            else
                error("Unknown distribution type $(typeof(d)).")
            end
        end
    end

    inparams_aggr_size = max_nedgeaggrs_per_input(pc)
    
    BitsProbCircuit(nodes, input_node_idxs, flatuplayers, FlatVector(downedges, downlayerends), 
                    down2upedges, input_node_vars, input_node_params, inparams_aggr_size, sumnode2edges)
end

const CuEdgeLayer = CuVector{BitsEdge}

struct CuBitsProbCircuit <: AbstractBitsProbCircuit
    
    nodes::CuVector{BitsNode}
    input_node_idxs::CuVector{UInt32}
    edge_layers_up::FlatVector{<:CuEdgeLayer}
    edge_layers_down::FlatVector{<:CuEdgeLayer}
    down2upedge::CuVector{Int32}
    input_node_vars::CuMatrix{UInt32}
    input_node_params::CuMatrix{Float32}
    inparams_aggr_size::Int32
    sumnode2edges::Dict{PlainProbCircuit,Tuple{Int,Int,Int}}

    CuBitsProbCircuit(bpc::BitsProbCircuit) = begin
        nodes = cu(bpc.nodes)
        input_node_idxs = cu(bpc.input_node_idxs)
        edge_layers_up = FlatVector(cu(bpc.edge_layers_up.vectors), 
                            bpc.edge_layers_up.ends)
        edge_layers_down = FlatVector(cu(bpc.edge_layers_down.vectors), 
                                      bpc.edge_layers_down.ends)
        down2upedge = cu(bpc.down2upedge)
        input_node_vars = cu(bpc.input_node_vars)
        input_node_params = cu(bpc.input_node_params)
        new(nodes, input_node_idxs, edge_layers_up, edge_layers_down, down2upedge, input_node_vars, 
            input_node_params, bpc.inparams_aggr_size, bpc.sumnode2edges)
    end
end

#####################
# constructor
#####################

function bit_circuit(pc::ProbCircuit; to_gpu::Bool = true)
    bpc = BitsProbCircuit(pc)
    if to_gpu
        CuBitsProbCircuit(bpc)
    else
        bpc
    end
end

#####################
# map parameters back to ProbCircuit
#####################

function copy_parameters!(pc::ProbCircuit, bpc::BitsProbCircuit)
    edge_layers = unflatten(bpc.edge_layers_up)
    input_node_params = bpc.input_node_params
    sumnode2edges = bpc.sumnode2edges
    foreach(pc) do n
        if n isa PlainInputNode
            glob_id = n.global_id
            if dist(n) isa Indicator
                nothing # do nothing
            elseif dist(n) isa BernoulliDist
                n.dist.logp = input_node_params[glob_id, 1]
            elseif dist(n) isa CategoricalDist
                ncats = length(n.dist.logps)
                n.dist.logps .= input_node_params[glob_id, 1:ncats]
            else
                error("Unknown distribution type $(typeof(dist(n))).")
            end
        elseif n isa PlainSumNode
            if length(n.params) > 1
                layer_id, edge_start_id, edge_end_id = sumnode2edges[n]
                @assert length(n.params) == edge_end_id - edge_start_id + 1
                for idx = 1 : length(n.params)
                    edge = edge_layers[layer_id][edge_start_id+idx-1]::SumEdge
                    n.params[idx] = edge.logp
                end
            end
        end
    end
    nothing
end
function copy_parameters!(pc::ProbCircuit, bpc::CuBitsProbCircuit)
    nodes = Array(bpc.nodes)
    input_node_idxs = Array(bpc.input_node_idxs)
    edge_layers_up = FlatVector(Array(bpc.edge_layers_up.vectors),
                                bpc.edge_layers_up.ends)
    edge_layers_down = FlatVector(Array(bpc.edge_layers_down.vectors),
                                bpc.edge_layers_down.ends)
    down2upedge = Array(bpc.down2upedge)
    input_node_vars = Array(bpc.input_node_vars)
    input_node_params = Array(bpc.input_node_params)
    bpc = BitsProbCircuit(nodes, input_node_idxs, edge_layers_up, edge_layers_down, down2upedge, 
                          input_node_vars, input_node_params, bpc.inparams_aggr_size, 
                          bpc.sumnode2edges)
                    
    copy_parameters!(pc, bpc)
end