using CUDA

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

abstract type AbstractBitsProbCircuit end

struct BitsProbCircuit <: AbstractBitsProbCircuit
    nodes::Vector{BitsNode}
    edge_layers_up::Vector{EdgeLayer}
    edge_layers_down::Vector{EdgeLayer}
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
    outedges = EdgeLayer[]
    uplayers = EdgeLayer[]

    add_node(bitsnode, layer_id) = begin
        # add node globally
        push!(nodes, bitsnode)
        push!(outedges, EdgeLayer())
        id = length(nodes)
        # add node to node layers 
        while length(node_layers) <= layer_id
            push!(node_layers, Int[])
        end
        push!(node_layers[layer_id+1], id)
        id
    end

    add_edge(parent_layer_id, edge, child_info) = begin
        # record up edges for upward pass
        while length(uplayers) < parent_layer_id
            push!(uplayers, EdgeLayer())
        end
        push!(uplayers[parent_layer_id], edge)

        # introduce invariant that primes are never at a lower layer than subs 
        if hassub(child_info) && child_info.prime_layer_id < child_info.sub_layer_id
            edge = rotate(edge)
        end

        # record out edges for downward pass
        push!(outedges[child_info.prime_id], edge)
        if hassub(child_info)
            push!(outedges[child_info.sub_id], edge)
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

    downlayers = EdgeLayer[]
    @assert length(node_layers[end]) == 1 && isempty(outedges[node_layers[end][1]])
    for node_layer in node_layers[end-1:-1:1]
        downlayer = EdgeLayer()
        for node_id in node_layer
            prime_edges = filter(e -> e.prime_id == node_id, outedges[node_id])
            partial = (length(prime_edges) != length(outedges[node_id]))
            for i in 1:length(prime_edges)
                edge = prime_edges[i]
                if edge.prime_id == node_id
                    tag = tag_index(i, length(prime_edges))
                    # tag whether this series of edges is partial or complete
                    partial && (tag |= (one(UInt8) << 2))
                    # tag whether this sub edge is the only outgoing edge from sub
                    if hassub(edge) && length(outedges[edge.sub_id]) == 1
                        tag |= (one(UInt8) << 3)
                    end
                    edge = changetag(edge, tag)
                    push!(downlayer, edge)
                end
            end
        end
        if !isempty(downlayer)
            push!(downlayers, downlayer)
        end
    end
    
    BitsProbCircuit(nodes, uplayers, downlayers)
end

const CuEdgeLayer = CuVector{BitsEdge}

struct CuBitsProbCircuit <: AbstractBitsProbCircuit
    nodes::CuVector{BitsNode}
    edge_layers_up::Vector{CuEdgeLayer}
    edge_layers_down::Vector{CuEdgeLayer}
    CuBitsProbCircuit(bpc::BitsProbCircuit) = begin
        nodes = cu(bpc.nodes)
        edge_layers_up = map(cu, bpc.edge_layers_up)
        edge_layers_down = map(cu, bpc.edge_layers_down)
        new(nodes, edge_layers_up, edge_layers_down)
    end
end

##################################################################################
# Init marginals
###################################################################################

function balance_threads(num_edges, num_examples, config; mine, maxe)
    # prefer to assign threads to examples, they do not require memory synchronization
    # make sure the number of example threads is a multiple of 32
    ex_threads = min(config.threads, cld(num_examples,32)*32)
    ex_blocks = cld(num_examples, ex_threads)
    edge_threads = config.threads ÷ ex_threads
    edge_blocks_min = cld(num_edges, edge_threads * maxe)
    edge_blocks_max = cld(num_edges, edge_threads * mine)
    edge_blocks_occupy = cld(config.blocks, ex_blocks)
    edge_blocks = min(max(edge_blocks_min, edge_blocks_occupy), edge_blocks_max)
    ((edge_threads, ex_threads), (edge_blocks, ex_blocks))
end

macro xrange(x_start, x_end, length)
    return quote
        blockdim = blockDim().x
        x_work::Int32 = cld($(esc(length))::Int32, (blockdim * gridDim().x))
        $(esc(x_start))::Int32 = ((blockIdx().x - one(Int32)) * blockdim + threadIdx().x - one(Int32)) * x_work + one(Int32)
        $(esc(x_end))::Int32 = min($(esc(x_start)) + x_work - one(Int32), $(esc(length))::Int32)       
    end
end

macro yindex(y_id)
    return quote
        $(esc(y_id))::Int32 = (blockIdx().y - one(Int32)) * blockDim().y + threadIdx().y
    end
end

function init_mar!_kernel(mars, nodes, data, example_ids)
    # this kernel follows the structure of the layer eval kernel, would probably be faster to have 1 thread process multiple examples, rather than multiple nodes 
    
    num_nodes::Int32 = length(nodes)
    @xrange node_start node_end num_nodes
    num_examples::Int32 = length(example_ids)
    @yindex ex_id

    # TODO can the first bounds check be dropped?
    @inbounds if ex_id <= num_examples
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
    @assert size(mars,1) >= length(example_ids)
    kernel = @cuda name="init_mar!" launch=false init_mar!_kernel(mars, bpc.nodes, data,example_ids) 
    config = launch_configuration(kernel.fun)
    threads, blocks = balance_threads(length(bpc.nodes), length(example_ids), config; mine, maxe)
    if debug
        println("Node initialization")
        println("  config=$config, threads=$threads, blocks=$blocks, nodes/thread=$(Float32(length(bpc.nodes)/threads[1]/blocks[1]))")
        CUDA.@time kernel(mars, bpc.nodes, data, example_ids; threads, blocks)
    else
        kernel(mars, bpc.nodes, data, example_ids; threads, blocks)
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

function layer_up_kernel(mars, layer, num_examples)

    num_edges = length(layer) % Int32
    @xrange edge_start edge_end num_edges
    @yindex ex_id

    @inbounds if ex_id <= num_examples

        local acc::Float32    
        owned_node::Bool = false
        
        for edge_id = edge_start:edge_end

            edge = layer[edge_id]

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

function layer_up(mars, bpc, layer_id, num_examples; mine, maxe, debug=false)
    layer = bpc.edge_layers_up[layer_id]
    kernel = @cuda name="layer_up" launch=false layer_up_kernel(mars, layer, num_examples) 
    config = launch_configuration(kernel.fun)
    threads, blocks = balance_threads(length(layer), num_examples, config; mine, maxe)
    if debug
        println("Layer $layer_id")
        println("  config=$config, threads=$threads, blocks=$blocks, edges/thread=$(Float32(length(layer)/threads[1]/blocks[1]))")
        CUDA.@time kernel(mars, layer; threads, blocks)
    else
        kernel(mars, layer, num_examples; threads, blocks)
    end
    nothing
end

# run entire circuit
function eval_circuit(mars, bpc, data, example_ids; mine, maxe, debug=false)
    init_mar!(mars, bpc, data, example_ids; mine, maxe, debug)
    for i in 1:length(bpc.edge_layers_up)
        layer_up(mars, bpc, i, length(example_ids); mine, maxe, debug)
    end
    nothing
end


##################################################################################
# Downward pass
##################################################################################

function layer_down_kernel(flows, _mars, layer, num_examples, node_aggr, edge_aggr)

    mars = Base.Experimental.Const(_mars)
    @yindex ex_id

    num_edges = length(layer) % Int32
    @xrange edge_start edge_end num_edges
    
    blockdimx = blockDim().x

    edge_work = cld(num_edges, (blockdimx * gridDim().x))
    edge_start = ((blockIdx().x - one(Int32)) * blockdimx + threadIdx().x - one(Int32)) * edge_work + one(Int32)
    edge_end = min(edge_start + edge_work - one(Int32), num_edges)

    block_edge_start = ((blockIdx().x - one(Int32)) * blockdimx) * edge_work + one(Int32)
    linear_threadidx = threadIdx().x + (threadIdx().y-one(Int32)) * gridDim().x        
 
    local edges_per_block::Int32

    if !isnothing(edge_aggr)
        edges_per_block = cld(num_edges, gridDim().x)
        shmem = CuDynamicSharedArray(Float32, (num_examples, edges_per_block))
    end

    @inbounds if ex_id <= num_examples

        local acc::Float32    
        local prime_mar::Float32

        owned_node::Bool = false
        
        for edge_id = edge_start:edge_end

            edge = layer[edge_id]

            parent_id = edge.parent_id
            prime_id = edge.prime_id
            sub_id = edge.sub_id

            tag = edge.tag
            firstedge = isfirst(tag)
            lastedge = islast(tag)
            issum = edge isa SumEdge

            if firstedge
                partial = ispartial(tag)
                owned_node = !partial
            end

            edge_flow = flows[ex_id, parent_id]
            if issum
                parent_mar = mars[ex_id, parent_id]
                child_prob = mars[ex_id, prime_id] + edge.logp
                if sub_id != 0
                    child_prob += mars[ex_id, sub_id]
                end
                edge_flow = edge_flow + child_prob - parent_mar
            end
                
            if !isnothing(edge_aggr) && isfinite(edge_flow)
                exp_edge_flow = @fastmath exp(edge_flow)
                shmem[ex_id, edge_id-block_edge_start+one(Int32)] = exp_edge_flow
            end    

            if sub_id != 0 
                if isonlysubedge(tag)
                    flows[ex_id, sub_id] = edge_flow
                else
                    CUDA.@atomic flows[ex_id, sub_id] = logsumexp(flows[ex_id, sub_id], edge_flow)
                end
            end

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

    @inbounds if !isnothing(edge_aggr)
        while num_examples > 1
            sync_threads()
            fold = cld(num_examples, 2)
            if num_examples >= ex_id > fold
                for edge_id = edge_start:edge_end
                    shmem[ex_id-fold, edge_id-block_edge_start+one(Int32)] += shmem[ex_id, edge_id-block_edge_start+one(Int32)]
                end
            end
            num_examples = cld(num_examples, 2)
        end
        sync_threads()
        if ex_id == 1
            for edge_id = edge_start:edge_end
                CUDA.@atomic edge_aggr[edge_id] += shmem[1, edge_id-block_edge_start+one(Int32)]
            end
        end
    end

    nothing
end

function layer_down_kernel3(flows, _mars, layer, num_ex_threads, num_examples, node_aggr, edge_aggr)

    mars = Base.Experimental.Const(_mars)
    
    num_edges = length(layer) % Int32
    
    threadid_block = threadIdx().x
    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadid_block 

    edge_batch, ex_id = fldmod1(threadid, num_ex_threads)

    grid_num_threads = gridDim().x * blockDim().x
    num_edge_batches = cld(grid_num_threads, num_ex_threads)
    edge_work = cld(num_edges, num_edge_batches)
    edge_start = (edge_batch-one(Int32))*edge_work + one(Int32)
    edge_end = min(edge_start + edge_work - one(Int32), num_edges)

    warp_lane = mod1(threadid_block, warpsize())
                
    local acc::Float32    
    local prime_mar::Float32

    owned_node::Bool = false
    
    @inbounds for edge_id = edge_start:edge_end

        edge = layer[edge_id]

        parent_id = edge.parent_id
        prime_id = edge.prime_id
        sub_id = edge.sub_id

        tag = edge.tag
        firstedge = isfirst(tag)
        lastedge = islast(tag)
        issum = edge isa SumEdge

        if firstedge
            partial = ispartial(tag)
            owned_node = !partial
        end

        if ex_id <= num_examples
            edge_flow = flows[ex_id, parent_id]
            if issum
                parent_mar = mars[ex_id, parent_id]
                child_prob = mars[ex_id, prime_id] + edge.logp
                if sub_id != 0
                    child_prob += mars[ex_id, sub_id]
                end
                edge_flow = edge_flow + child_prob - parent_mar
            end
        else
            edge_flow = -Inf32
        end
        
        # make sure this is run on all warp threads, regardless of `ex_id`
        if !isnothing(edge_aggr)
            exp_edge_flow = exp(edge_flow)
            exp_edge_flow = CUDA.reduce_warp(+, exp_edge_flow)
            if warp_lane == 1
                CUDA.@atomic edge_aggr[edge_id] += exp_edge_flow
            end
        end    

        if ex_id <= num_examples

            if sub_id != 0 
                if isonlysubedge(tag)
                    flows[ex_id, sub_id] = edge_flow
                else
                    CUDA.@atomic flows[ex_id, sub_id] = logsumexp(flows[ex_id, sub_id], edge_flow)
                end
            end

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

function layer_down(flows, mars, bpc, layer_id, num_examples, node_aggr, edge_aggr; mine, maxe, debug=false)
    layer = bpc.edge_layers_down[layer_id]
    kernel = @cuda name="layer_down" launch=false layer_down_kernel(flows, mars, layer, num_examples, node_aggr, edge_aggr) 
    config = launch_configuration(kernel.fun)
    threads, blocks = balance_threads(length(layer), num_examples, config; mine, maxe)
    edges_per_block = cld(length(layer), blocks[1])
    shmem = isnothing(edge_aggr) ? 0 : num_examples*edges_per_block*sizeof(Float32)
    if debug
        println("Layer $layer_id")
        println("  config=$config, threads=$threads, blocks=$blocks, edges/thread=$(Float32(length(layer)/threads[1]/blocks[1]))")
        CUDA.@time kernel(flows, mars, layer; threads, blocks)
    else
        kernel(flows, mars, layer, num_examples, node_aggr, edge_aggr; threads, blocks, shmem)
    end
    nothing
end

function layer_down2(flows, mars, bpc, layer_id, num_examples, node_aggr, edge_aggr; mine, maxe, debug=false)
    layer = bpc.edge_layers_down[layer_id]
    kernel = @cuda name="layer_down3" launch=false layer_down_kernel3(flows, mars, layer, Int32(32), Int32(num_examples), node_aggr, edge_aggr) 
    config = launch_configuration(kernel.fun)
    threads, blocks = balance_threads(length(layer), num_examples, config; mine, maxe)
    num_example_threads = threads[2]
    if debug
        println("Layer $layer_id")
        println("  config=$config, threads=$threads, blocks=$blocks, edges/thread=$(Float32(length(layer)/threads[1]/blocks[1]))")
        CUDA.@time kernel(flows, mars, layer; threads, blocks)
    else
        kernel(flows, mars, layer, Int32(num_example_threads), Int32(num_examples), node_aggr, edge_aggr; 
                threads = threads[1]*threads[2], blocks = blocks[1]*blocks[2])
    end
    nothing
end

# run entire circuit
function flows_circuit(flows, mars, bpc, num_examples, node_aggr=nothing, edge_aggrs=nothing; mine, maxe, debug=false)
    if debug
        println("Initializing flows")
        CUDA.@time CUDA.@sync begin 
            flows .= -Inf32
            flows[:,end] .= zero(Float32)
        end
    else
        flows .= -Inf32
        flows[:,end] .= zero(Float32)
    end
    for i in 1:length(bpc.edge_layers_down)
        edge_aggr = isnothing(edge_aggrs) ? nothing : edge_aggrs[i]
        layer_down(flows, mars, bpc, i, num_examples, node_aggr, edge_aggr; mine, maxe, debug)
    end
    nothing
end

function flows_circuit2(flows, mars, bpc, num_examples, node_aggr=nothing, edge_aggrs=nothing; mine, maxe, debug=false)
    if debug
        println("Initializing flows")
        CUDA.@time CUDA.@sync begin 
            flows .= -Inf32
            flows[:,end] .= zero(Float32)
        end
    else
        flows .= -Inf32
        flows[:,end] .= zero(Float32)
    end
    for i in 1:length(bpc.edge_layers_down)
        edge_aggr = isnothing(edge_aggrs) ? nothing : edge_aggrs[i]
        layer_down2(flows, mars, bpc, i, num_examples, node_aggr, edge_aggr; mine, maxe, debug)
    end
    nothing
end

function probs_flows_circuit(flows, mars, bpc, data, example_ids; mine, maxe, debug=false)
    eval_circuit(mars, bpc, data, example_ids; mine, maxe, debug)
    flows_circuit(flows, mars, bpc, length(example_ids); mine, maxe, debug)
    nothing
end


#######################
### Full Epoch Marginal
######################

function loglikelihood(data::CuArray, bpc::CuBitsProbCircuit; 
            batch_size=512, mars_mem = nothing, 
            mine=2, maxe=32, debug=false)
    
    num_examples = size(data)[1]
    num_nodes = length(bpc.nodes)
    
    marginals = if isnothing(mars_mem)
        CuMatrix{Float32}(undef, batch_size, num_nodes)
    else
        @assert size(mars_mem, 1) >= batch_size
        @assert size(mars_mem, 2) == num_nodes
        mars_mem
    end

    log_likelihood = zero(Float64)

    for batch_start = 1:batch_size:num_examples

        batch_end = min(batch_start+batch_size-1, num_examples)
        batch = batch_start:batch_end

        eval_circuit(marginals, bpc, data, batch; mine, maxe, debug)

        log_likelihood += sum(marginals[:,end])        
    end
    return log_likelihood / num_examples
end