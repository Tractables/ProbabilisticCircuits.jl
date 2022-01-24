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

##################################################################################
# Init marginals
###################################################################################

function balance_threads(num_items, num_examples, config; mine, maxe, contiguous_warps=true)
    block_threads = config.threads
    # make sure the number of example threads is a multiple of 32
    example_threads = contiguous_warps ? (cld(num_examples,32) * 32) : num_examples
    num_item_batches = cld(num_items, maxe)
    num_blocks = cld(num_item_batches * example_threads, block_threads)
    if num_blocks < config.blocks
        max_num_item_batch = cld(num_items, mine)
        max_num_blocks = cld(max_num_item_batch * example_threads, block_threads)
        num_blocks = min(config.blocks, max_num_blocks)
        num_item_batches = (num_blocks * block_threads) ÷ example_threads
    end
    item_work = cld(num_items, num_item_batches)
    @assert item_work*block_threads*num_blocks >= example_threads*num_items
    block_threads, num_blocks, example_threads, item_work
end

function init_mar!_kernel(mars, nodes, data, example_ids,
            num_ex_threads::Int32, node_work::Int32)
    # this kernel follows the structure of the layer eval kernel, would probably be faster to have 1 thread process multiple examples, rather than multiple nodes 
    
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work 
    node_end = min(node_start + node_work - one(Int32), length(nodes))

    @inbounds if ex_id <= length(example_ids)
        for node_id = node_start:node_end

            node = nodes[node_id]
            
            mars[ex_id, node_id] = 
                if (node isa BitsInnerNode)
                    node.issum ? -Inf32 : zero(Float32)
                else
                    orig_ex_id::Int32 = example_ids[ex_id]
                    leaf = node::BitsLiteral
                    lit = leaf.literal
                    v = data[orig_ex_id, abs(lit)]
                    if ismissing(v)
                        zero(Float32)
                    elseif (lit > 0) == v
                        zero(Float32)
                    else
                        -Inf32
                    end
                end
        end
    end
    nothing
end

function init_mar!(mars, bpc, data, example_ids; mine, maxe, debug=false)
    num_examples = length(example_ids)
    num_nodes = length(bpc.nodes)
    
    dummy_args = (mars, bpc.nodes, data, example_ids, Int32(1), Int32(1))
    kernel = @cuda name="init_mar!" launch=false init_mar!_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_nodes, num_examples, config; mine, maxe)
    
    args = (mars, bpc.nodes, data, example_ids, 
            Int32(num_example_threads), Int32(node_work))
    if debug
        println("Node initialization")
        @show threads blocks num_example_threads node_work num_nodes num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

##################################################################################
# Upward pass
##################################################################################

import StatsFuns: logsumexp #extend

function logsumexp(x::Float32,y::Float32)
    if isfinite(x) && isfinite(y)
        # note: @fastmath does not work with infinite values, so do not apply above
        @fastmath max(x,y) + log1p(exp(-abs(x-y))) 
    else
        max(x,y)
    end
end

function layer_up_kernel(mars, edges, 
            num_ex_threads::Int32, num_examples::Int32, 
            layer_start::Int32, edge_work::Int32, layer_end::Int32)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch - one(Int32)) * edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end)

    @inbounds if ex_id <= num_examples

        local acc::Float32    
        owned_node::Bool = false
        
        for edge_id = edge_start:edge_end

            edge = edges[edge_id]

            tag = edge.tag
            isfirstedge = isfirst(tag)
            islastedge = islast(tag)
            issum = edge isa SumEdge
            owned_node |= isfirstedge

            # compute probability coming from child
            child_prob = mars[ex_id, edge.prime_id]
            if edge.sub_id != 0
                child_prob += mars[ex_id, edge.sub_id]
            end
            if issum
                child_prob += edge.logp
            end

            # accumulate probability from child
            if isfirstedge || (edge_id == edge_start)  
                acc = child_prob
            elseif issum
                acc = logsumexp(acc, child_prob)
            else
                acc += child_prob
            end

            # write to global memory
            if islastedge || (edge_id == edge_end)   
                pid = edge.parent_id

                if islastedge && owned_node
                    # no one else is writing to this global memory
                    mars[ex_id, pid] = acc
                else
                    if issum
                        CUDA.@atomic mars[ex_id, pid] = logsumexp(mars[ex_id, pid], acc)
                    else
                        CUDA.@atomic mars[ex_id, pid] += acc
                    end 
                end    
            end
        end
    end
    nothing
end

function layer_up(mars, bpc, layer_start, layer_end, num_examples; mine, maxe, debug=false)
    edges = bpc.edge_layers_up.vectors
    num_edges = layer_end - layer_start + 1
    dummy_args = (mars, edges, 
                  Int32(32), Int32(num_examples), 
                  Int32(1), Int32(1), Int32(2))
    kernel = @cuda name="layer_up" launch=false layer_up_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe)
    
    args = (mars, edges, 
            Int32(num_example_threads), Int32(num_examples), 
            Int32(layer_start), Int32(edge_work), Int32(layer_end))
    if debug
        println("Layer $layer_start:$layer_end")
        @show num_edges num_examples threads blocks num_example_threads edge_work
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

# run entire circuit
function eval_circuit(mars, bpc, data, example_ids; mine, maxe, debug=false)
    init_mar!(mars, bpc, data, example_ids; mine, maxe, debug)
    layer_start = 1
    for layer_end in bpc.edge_layers_up.ends
        layer_up(mars, bpc, layer_start, layer_end, length(example_ids); mine, maxe, debug)
        layer_start = layer_end + 1
    end
    nothing
end

#######################
### Full Epoch Likelihood
######################

function prep_memory(reuse, sizes, exact = map(x -> true, sizes))
    if isnothing(reuse)
        return CuArray{Float32}(undef, sizes...)
    else
        @assert ndims(reuse) == length(sizes)
        for d = 1:length(sizes)
            if exact[d]
                @assert size(reuse, d) == sizes[d] 
            else
                @assert size(reuse, d) >= sizes[d] 
            end
        end
        return reuse
    end
end

function loglikelihood(data::CuArray, bpc::CuBitsProbCircuit; 
    batch_size, mars_mem = nothing, 
    mine=2, maxe=32, debug=false)

    num_examples = size(data)[1]
    num_nodes = length(bpc.nodes)
    num_batches = cld(num_examples, batch_size)

    marginals = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))

    log_likelihoods = CUDA.zeros(Float32, num_batches, 1)

    batch_index = 0
    for batch_start = 1:batch_size:num_examples

        batch_end = min(batch_start+batch_size-1, num_examples)
        batch = batch_start:batch_end
        num_batch_examples = length(batch)
        batch_index += 1

        eval_circuit(marginals, bpc, data, batch; mine, maxe, debug)
        
        @views sum!(
            log_likelihoods[batch_index:batch_index, 1:1], 
            marginals[1:num_batch_examples,end:end])
    end

    return sum(log_likelihoods) / num_examples
end

##################################################################################
# Downward pass
##################################################################################

function layer_down_kernel(flows, edge_aggr, edges, _mars,
            num_ex_threads::Int32, num_examples::Int32, 
            layer_start::Int32, edge_work::Int32, layer_end::Int32)

    mars = Base.Experimental.Const(_mars)
        
    threadid_block = threadIdx().x
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadid_block 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    edge_start = layer_start + (edge_batch-one(Int32))*edge_work 
    edge_end = min(edge_start + edge_work - one(Int32), layer_end) 
   
    warp_lane = mod1(threadid_block, warpsize())

    local acc::Float32    
    local prime_mar::Float32

    owned_node::Bool = false
    
    @inbounds for edge_id = edge_start:edge_end

        edge = edges[edge_id]

        parent_id = edge.parent_id
        prime_id = edge.prime_id
        sub_id = edge.sub_id

        tag = edge.tag
        firstedge = isfirst(tag)
        lastedge = islast(tag)
        issum = edge isa SumEdge
        active = (ex_id <= num_examples)
        
        if firstedge
            partial = ispartial(tag)
            owned_node = !partial
        end

        if active
            
            edge_flow = flows[ex_id, parent_id]

            if issum
                parent_mar = mars[ex_id, parent_id]
                child_prob = mars[ex_id, prime_id] + edge.logp
                if sub_id != 0
                    child_prob += mars[ex_id, sub_id]
                end
                edge_flow = edge_flow + child_prob - parent_mar
            end

            if sub_id != 0 
                if isonlysubedge(tag)
                    flows[ex_id, sub_id] = edge_flow
                else
                    CUDA.@atomic flows[ex_id, sub_id] = logsumexp(flows[ex_id, sub_id], edge_flow)
                end            
            end
        end
        
        # make sure this is run on all warp threads, regardless of `active`
        if !isnothing(edge_aggr)
            exp_edge_flow = active ? exp(edge_flow) : zero(Float32)
            exp_edge_flow = CUDA.reduce_warp(+, exp_edge_flow)
            if warp_lane == 1
                CUDA.@atomic edge_aggr[edge_id] += exp_edge_flow
            end
        end

        if active

            # accumulate flows from parents
            if firstedge || (edge_id == edge_start)  
                acc = edge_flow
            else
                acc = logsumexp(acc, edge_flow)
            end

            # write to global memory
            if lastedge || (edge_id == edge_end)   
                if lastedge && owned_node
                    # no one else is writing to this global memory
                    flows[ex_id, prime_id] = acc
                else
                    CUDA.@atomic flows[ex_id, prime_id] = logsumexp(flows[ex_id, prime_id], acc)
                end
            end
        end
        
    end

    nothing
end

function layer_down(flows, edge_aggr, bpc, mars, 
                    layer_start, layer_end, num_examples; 
                    mine, maxe, debug=false)
    edges = bpc.edge_layers_down.vectors
    num_edges = layer_end-layer_start+1
    dummy_args = (flows, edge_aggr, edges, mars, 
                  Int32(32), Int32(num_examples), 
                  Int32(1), Int32(1), Int32(2))
    kernel = @cuda name="layer_down" launch=false layer_down_kernel(dummy_args...) 
    config = launch_configuration(kernel.fun)

    # configure thread/block balancing
    threads, blocks, num_example_threads, edge_work = 
        balance_threads(num_edges, num_examples, config; mine, maxe, contiguous_warps=true)
    
    args = (flows, edge_aggr, edges, mars, 
            Int32(num_example_threads), Int32(num_examples), 
            Int32(layer_start), Int32(edge_work), Int32(layer_end))
    if debug
        println("Layer $layer_start:$layer_end")
        @show threads blocks num_example_threads edge_work, num_edges num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end
    nothing
end

function flows_circuit(flows, edge_aggr, bpc, mars, num_examples; mine, maxe, debug=false)
    init_flows() = begin 
        flows .= -Inf32
        flows[:,end] .= zero(Float32)
    end
    if debug
        println("Initializing flows")
        CUDA.@time CUDA.@sync init_flows()
    else
        init_flows()
    end

    layer_start = 1
    for layer_end in bpc.edge_layers_down.ends
        layer_down(flows, edge_aggr, bpc, mars, 
                   layer_start, layer_end, num_examples; 
                   mine, maxe, debug)
        layer_start = layer_end + 1
    end
    nothing
end

function probs_flows_circuit(flows, mars, edge_aggr, bpc, data, example_ids; mine, maxe, debug=false)
    eval_circuit(mars, bpc, data, example_ids; mine, maxe, debug)
    flows_circuit(flows, edge_aggr, bpc, mars, length(example_ids); mine, maxe, debug)
    nothing
end

##################################################################################
# Count siblings
##################################################################################

function count_siblings_kernel(node_aggr, edges)
    edge_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 
    @inbounds if edge_id <= length(edges)
        edge = edges[edge_id]
        if edge isa SumEdge
            parent_id = edge.parent_id
            CUDA.@atomic node_aggr[parent_id] += one(Float32)
        end
    end      
    nothing
end

function count_siblings(node_aggr, bpc)
    # reset aggregates
    node_aggr .= zero(Float32)
    edges = bpc.edge_layers_down.vectors
    args = (node_aggr, edges)
    kernel = @cuda name="count_siblings" launch=false count_siblings_kernel(args...) 
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges), threads)
    kernel(args...; threads, blocks)
    nothing
end

##################################################################################
# Pseudocounts
##################################################################################

function add_pseudocount_kernel(edge_aggr, edges, _node_aggr, pseudocount)
    node_aggr = Base.Experimental.Const(_node_aggr)
    edge_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 
    @inbounds if edge_id <= length(edges)
        edge = edges[edge_id]
        if edge isa SumEdge
            parent_id = edge.parent_id
            CUDA.@atomic edge_aggr[edge_id] += pseudocount / node_aggr[parent_id]
        end
    end      
    nothing
end

function add_pseudocount(edge_aggr, node_aggr, bpc, pseudocount)
    count_siblings(node_aggr, bpc)
    edges = bpc.edge_layers_down.vectors
    args = (edge_aggr, edges, node_aggr, Float32(pseudocount))
    kernel = @cuda name="add_pseudocount" launch=false add_pseudocount_kernel(args...) 
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges), threads)
    kernel(args...; threads, blocks)
    nothing
end

##################################################################################
# Aggregate node flows
##################################################################################

function aggr_node_flows_kernel(node_aggr, edges, _edge_aggr)
    edge_aggr = Base.Experimental.Const(_edge_aggr)
    edge_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if edge_id <= length(edges)
        edge = edges[edge_id]
        if edge isa SumEdge
            parent_id = edge.parent_id
            edge_flow = edge_aggr[edge_id]
            CUDA.@atomic node_aggr[parent_id] += edge_flow
        end
    end      
    nothing
end

function aggr_node_flows(node_aggr, bpc, edge_aggr)
    # reset aggregates
    node_aggr .= zero(Float32)
    edges = bpc.edge_layers_down.vectors
    args = (node_aggr, edges, edge_aggr)
    kernel = @cuda name="aggr_node_flows" launch=false aggr_node_flows_kernel(args...) 
    config = launch_configuration(kernel.fun)
    threads = config.threads
    blocks = cld(length(edges), threads)
    kernel(args...; threads, blocks)
    nothing
end

##################################################################################
# Update parameters
##################################################################################

function update_params_kernel(edges_down, edges_up, _down2upedge, _node_aggr, _edge_aggr, inertia)
    node_aggr = Base.Experimental.Const(_node_aggr)
    edge_aggr = Base.Experimental.Const(_edge_aggr)
    down2upedge = Base.Experimental.Const(_down2upedge)
    
    edge_id_down = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if edge_id_down <= length(edges_down)
        edge_down = edges_down[edge_id_down]
        if edge_down isa SumEdge 
            
            edge_id_up = down2upedge[edge_id_down]
            # only difference is the tag
            edge_up_tag = edges_up[edge_id_up].tag

            if !(isfirst(edge_up_tag) && islast(edge_up_tag))
                parent_id = edge_down.parent_id
                parent_flow = node_aggr[parent_id]
                edge_flow = edge_aggr[edge_id_down]

                old = inertia * exp(edge_down.logp)
                new = (one(Float32) - inertia) * edge_flow / parent_flow 
                new_log_param = log(old + new)

                edges_down[edge_id_down] = 
                    SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id, 
                            new_log_param, edge_down.tag)

                edges_up[edge_id_up] = 
                    SumEdge(parent_id, edge_down.prime_id, edge_down.sub_id, 
                            new_log_param, edge_up_tag)
            end
        end
    end      
    nothing
end

function update_params(bpc, node_aggr, edge_aggr; inertia = 0)
    edges_down = bpc.edge_layers_down.vectors
    edges_up = bpc.edge_layers_up.vectors
    down2upedge = bpc.down2upedge
    @assert length(edges_down) == length(down2upedge) == length(edges_up)

    args = (edges_down, edges_up, down2upedge, node_aggr, edge_aggr, Float32(inertia))
    kernel = @cuda name="update_params" launch=false update_params_kernel(args...) 
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(length(edges_down), threads)
    
    kernel(args...; threads, blocks)
    nothing
end

#######################
### Full-Batch EM
######################

function full_batch_em_step(bpc::CuBitsProbCircuit, data::CuArray; 
    batch_size, pseudocount, report_ll=true,
    mars_mem, flows_mem, node_aggr_mem, edge_aggr_mem,
    mine, maxe, debug)

    num_examples = size(data)[1]
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    num_batches = cld(num_examples, batch_size)

    marginals = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    node_aggr = prep_memory(node_aggr_mem, (num_nodes,))
    edge_aggr = prep_memory(edge_aggr_mem, (num_edges,))
    
    if report_ll 
        log_likelihoods = CUDA.zeros(Float32, num_batches, 1)
    end
    
    edge_aggr.= zero(Float32)
    batch_index = 0

    for batch_start = 1:batch_size:num_examples

        batch_end = min(batch_start+batch_size-1, num_examples)
        batch = batch_start:batch_end
        num_batch_examples = batch_end - batch_start + 1
        batch_index += 1

        probs_flows_circuit(flows, marginals, edge_aggr, bpc, data, batch; 
                            mine, maxe, debug)

        if report_ll
            @views sum!(
                log_likelihoods[batch_index:batch_index, 1:1], 
                marginals[1:num_batch_examples,end:end])
        end
    end
    add_pseudocount(edge_aggr, node_aggr, bpc, pseudocount)
    aggr_node_flows(node_aggr, bpc, edge_aggr)
    update_params(bpc, node_aggr, edge_aggr)

    return report_ll ? sum(log_likelihoods) / num_examples : 0.0
end

function full_batch_em(bpc::CuBitsProbCircuit, data::CuArray, num_epochs; 
    batch_size, pseudocount, report_ll=true,
    mars_mem = nothing, flows_mem = nothing, node_aggr_mem = nothing, edge_aggr_mem=nothing,
    mine=2, maxe=32, debug=false)

    for epoch = 1:num_epochs
        log_likelihood = full_batch_em_step(bpc, data; 
            batch_size, pseudocount, report_ll,
            mars_mem, flows_mem, node_aggr_mem, edge_aggr_mem,
            mine, maxe, debug)
        println("Full-batch EM epoch $epoch; train LL $log_likelihood")
    end
    
end

#######################
### Mini-Batch EM
######################

function mini_batch_em(bpc::CuBitsProbCircuit, data::CuArray, num_epochs; 
    batch_size, pseudocount, param_inertia, flow_memory, shuffle=:each_epoch,  
    mars_mem = nothing, flows_mem = nothing, node_aggr_mem = nothing, edge_aggr_mem=nothing,
    mine=2, maxe=32, debug=false)

    @assert pseudocount >= 0
    @assert 0 <= param_inertia < 1
    @assert 0 <= flow_memory  
    @assert shuffle ∈ [:never, :once, :each_epoch, :each_batch]

    num_examples = size(data)[1]
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    num_batches = num_examples ÷ batch_size # drop last incomplete batch

    @assert batch_size <= num_examples
    
    marginals = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    node_aggr = prep_memory(node_aggr_mem, (num_nodes,))
    edge_aggr = prep_memory(edge_aggr_mem, (num_edges,))
    
    edge_aggr .= zero(Float32)

    output_layer = @view marginals[1:batch_size,end]

    shuffled_indices_cpu = Vector{Int32}(undef, num_examples)
    shuffled_indices = CuVector{Int32}(undef, num_examples)
    batches = [@view shuffled_indices[1+(b-1)*batch_size : b*batch_size]
                for b in 1:num_batches]

    do_shuffle() = begin
        randperm!(shuffled_indices_cpu)
        copyto!(shuffled_indices, shuffled_indices_cpu)
    end

    (shuffle == :once) && do_shuffle()


    for epoch in 1:num_epochs

        log_likelihood = zero(Float32)

        (shuffle == :each_epoch) && do_shuffle()

        for batch in batches

            (shuffle == :each_batch) && do_shuffle()

            if iszero(flow_memory)
                edge_aggr .= zero(Float32)
            else
                # slowly forget old edge aggregates
                edge_aggr .*= one(Float32) - (batch_size + pseudocount) / flow_memory
            end

            probs_flows_circuit(flows, marginals, edge_aggr, bpc, data, batch; 
                                mine, maxe, debug)


            add_pseudocount(edge_aggr, node_aggr, bpc, pseudocount)
            aggr_node_flows(node_aggr, bpc, edge_aggr)
            update_params(bpc, node_aggr, edge_aggr; inertia = param_inertia)

            log_likelihood += @views sum(output_layer) / batch_size
        end
            
        log_likelihood /= num_batches
        total_flow = CUDA.@allowscalar node_aggr[end]
        println("Mini-batch EM iter $epoch; flows $total_flow; train LL $log_likelihood")
    end

    nothing
end
