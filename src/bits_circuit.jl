using CUDA

export copy_parameters!

###############################################
# bits representation of nodes
###############################################

abstract type BitsNode end

struct BitsInput{D} <: BitsNode
    variable::UInt32 # for now assume single variable
    dist::D
end

function bits(in::PlainInputNode, heap)
    vars = UInt32.(randvars(in))
    bits_dist = bits(dist(in), heap)
    BitsInput(vars..., bits_dist)
end

abstract type BitsInnerNode <: BitsNode end

struct BitsSum <: BitsInnerNode end
struct BitsProd <: BitsInnerNode end

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

@inline tagged_at(tag, i) = ((tag & one(tag) << i) != zero(tag))
@inline isfirst(tag) = tagged_at(tag, 0)
@inline islast(tag) = tagged_at(tag, 1)
@inline ispartial(tag) = tagged_at(tag, 2)
@inline isonlysubedge(tag) = tagged_at(tag, 3)

function tag_firstlast(i,n)
    tag = zero(UInt8)
    (i==1) && (tag |= one(UInt8))
    (i==n) && (tag |= (one(UInt8) << 1)) 
    tag
end

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

import CUDA: cu #extend

cu(fv::FlatVectors) =
    FlatVectors(cu(fv.vectors), fv.ends)

###############################################
# bits representation of circuit
###############################################

struct BitsProbCircuit
    
    # all the nodes in the circuit
    nodes::Vector{BitsNode}

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

    BitsProbCircuit(n, in, e1, e2, d, heap) = begin
        @assert length(n) >= length(in) > 0
        @assert length(e1.vectors) == length(e2.vectors)
        @assert allunique(e1.ends) "No empty layers allowed"
        @assert allunique(e2.ends) "No empty layers allowed"
        new(n, in, e1, e2, d, heap)
    end
end

struct CuBitsProbCircuit{BitsNodes <: BitsNode}
    
    # all the nodes in the circuit
    nodes::CuVector{BitsNodes}

    # the ids of the subset of nodes that are inputs
    input_node_ids::CuVector{UInt32}
    
    # layers of edges for upward pass
    edge_layers_up::FlatVectors{CuVector{BitsEdge}}
    
    # layers of edges for downward pass
    edge_layers_down::FlatVectors{CuVector{BitsEdge}}
    
    # mapping from downward pass edge id to upward pass edge id
    down2upedge::CuVector{Int32}
    
    # memory used by input nodes for their parameters and parameter learning
    heap::CuVector{Float32}

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
        new{BitsNodes}(nodes, input_node_ids, edge_layers_up, edge_layers_down, down2upedge, heap)
    end

end

###############################################
# converting a PC into a BitsPC
###############################################

struct NodeInfo
    prime_id::Int
    prime_layer_id::Int
    sub_id::Int # 0 means no sub
    sub_layer_id::Int
end

struct InputsInfo
    layer_id::Int
    edge_start_id::Int
    edge_end_id::Int
end

struct OutputInfo
    edge::BitsEdge
    parent_layer_id::Int
    id_within_uplayer::Int
end

max_layer_id(ni::NodeInfo) = max(ni.prime_layer_id, ni.sub_layer_id)

function BitsProbCircuit(pc::ProbCircuit; eager_materialize=true, collapse_elements=true)

    nodes = BitsNode[]
    input_node_ids = UInt32[]
    node_layers = Vector{Int}[]
    outputs = Vector{OutputInfo}[]
    uplayers = Vector{BitsEdge}[]
    heap = Float32[]
    
    sumnode2edges = Dict{ProbCircuit,InputsInfo}()

    add_node(bitsnode, layer_id) = begin
        # add node globally
        push!(nodes, bitsnode)
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
        @assert uplayers[outputinfo[2]][outputinfo[3]] == edge

        # TODO should be edge.prime_id? because it can be rotated?
        push!(outputs[edge.prime_id], outputinfo)
        if hassub(edge)
            push!(outputs[edge.sub_id], outputinfo)
        end
    end

    f_leaf(node) = begin
        node_id = add_node(bits(node, heap), 0)
        NodeInfo(node_id, 0, 0, 0)
    end
    
    # TODO
    f_inner(node, children_info) = begin
        if (length(children_info) == 1 #TODO check condition
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
            layer_id = maximum(max_layer_id, children_info) + 1
            # TODO
            node_id = add_node(BitsInnerNode(issum(node)), layer_id)
            if issum(node)
                for i = 1:length(children_info)
                    logp = node.params[i]
                    child_info = children_info[i]
                    tag = tag_firstlast(i, length(children_info))
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
                    tag = tag_firstlast(i, length(double_infos))
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
        edge = MulEdge(node_id, root_info.prime_id, root_info.sub_id, tag_firstlast(1,1))
        add_edge(layer_id, edge, root_info)
    end

    flatuplayers = FlatVectors(uplayers)

    downedges = EdgeLayer()
    downlayerends = Int[]
    down2upedges = Int32[]
    uplayerstarts = [1, map(x -> x+1, flatuplayers.ends[1:end-1])...]
    @assert length(node_layers[end]) == 1 && isempty(outputs[node_layers[end][1]])
    for node_layer in node_layers[end-1:-1:1]
        for node_id in node_layer
            prime_edges = filter(e -> e[1].prime_id == node_id, outputs[node_id])
            partial = (length(prime_edges) != length(outputs[node_id]))
            for i in 1:length(prime_edges)
                edge = prime_edges[i][1]
                if edge.prime_id == node_id
                    # record the index in flatuplayers corresponding to this downedge
                    upedgeindex = uplayerstarts[prime_edges[i][2]] + prime_edges[i][3] - 1
                    push!(down2upedges, upedgeindex)

                    # update the tag and record down edge
                    tag = tag_firstlast(i, length(prime_edges))
                    # tag whether this series of edges is partial or complete
                    partial && (tag |= (one(UInt8) << 2))
                    # tag whether this sub edge is the only outgoing edge from sub
                    if hassub(edge) && length(outputs[edge.sub_id]) == 1
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
    
    BitsProbCircuit(nodes, flatuplayers, FlatVectors(downedges, downlayerends), 
                    down2upedges, input_node_params)
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
    edge_layers_up = FlatVectors(Array(bpc.edge_layers_up.vectors),
                                bpc.edge_layers_up.ends)
    edge_layers_down = FlatVectors(Array(bpc.edge_layers_down.vectors),
                                bpc.edge_layers_down.ends)
    down2upedge = Array(bpc.down2upedge)
    input_node_vars = Array(bpc.input_node_vars)
    input_node_params = Array(bpc.input_node_params)
    bpc = BitsProbCircuit(nodes, input_node_idxs, edge_layers_up, edge_layers_down, down2upedge, 
                          input_node_vars, input_node_params, bpc.inparams_aggr_size, 
                          bpc.sumnode2edges)
                    
    copy_parameters!(pc, bpc)
end