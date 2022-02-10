using CUDA, Random

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
                edge_flow = edge_flow * exp(child_prob - parent_mar)
            end

            if sub_id != 0 
                if isonlysubedge(tag)
                    flows[ex_id, sub_id] = edge_flow
                else
                    CUDA.@atomic flows[ex_id, sub_id] += edge_flow
                end            
            end
        end
        
        # make sure this is run on all warp threads, regardless of `active`
        if !isnothing(edge_aggr)
            !active && (edge_flow = zero(Float32))
            edge_flow_warp = CUDA.reduce_warp(+, edge_flow)
            if warp_lane == 1
                CUDA.@atomic edge_aggr[edge_id] += edge_flow_warp
            end
        end

        if active

            # accumulate flows from parents
            if firstedge || (edge_id == edge_start)  
                acc = edge_flow
            else
                acc += edge_flow
            end

            # write to global memory
            if lastedge || (edge_id == edge_end)   
                if lastedge && owned_node
                    # no one else is writing to this global memory
                    flows[ex_id, prime_id] = acc
                else
                    CUDA.@atomic flows[ex_id, prime_id] += acc
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
        flows .= zero(Float32)
        flows[:,end] .= one(Float32)
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

##################################################################################
# Downward pass for input nodes
##################################################################################

function input_flows_circuit_kernel(flows, nodes, inparams_aggr, input_node_idxs, input_node_vars, input_node_params,
                                    mars, data, example_ids, num_ex_threads::Int32, node_work::Int32)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    node_batch, ex_id = fldmod1(threadid, num_ex_threads)

    node_start = one(Int32) + (node_batch - one(Int32)) * node_work
    node_end = min(node_start + node_work - one(Int32), length(input_node_idxs))

    @inbounds if ex_id <= length(example_ids)
        for node_id = node_start : node_end
            orig_ex_id::Int32 = example_ids[ex_id]

            orig_node_id::UInt32 = input_node_idxs[node_id]
            node_flow::Float32 = flows[orig_ex_id, orig_node_id]
            node::BitsInputNode = nodes[orig_node_id]
            glob_id = node.global_id::UInt32
            dist_type = leaf.dist_type::UInt8

            if dist_type == UInt8(1) # LiteralDist
                nothing # need to do nothing
            elseif dist_type == UInt8(2) # BernoulliDist
                var = input_node_vars[glob_id, 1]::UInt32
                v = data[orig_ex_id, var]
                if !ismissing(v)
                    if v # v == true
                        CUDA.@atomic inparams_aggr[glob_id, 1] += node_flow
                    else # v == false
                        CUDA.@atomic inparams_aggr[glob_id, 2] += node_flow
                    end
                end
            elseif dist_type == UInt8(3) # CategoricalDist
                var = input_node_vars[glob_id, 1]::UInt32
                v = data[orig_ex_id, var]
                if !ismissing(v)
                    CUDA.@atomic inparams_aggr[glob_id, v] += node_flow
                end
            else
                @assert false
            end

        end
    end

    nothing
end

function input_flows_circuit(flows, inparams_aggr, bpc, mars, data, example_ids; mine, maxe, debug=false)
    input_node_idxs = bpc.input_node_idxs
    input_node_vars = bpc.input_node_vars
    input_node_params = bpc.input_node_params

    num_examples = length(example_ids)
    num_input_nodes = length(input_node_idxs)

    dummy_args = (flows, bpc.nodes, inparams_aggr, input_node_idxs, input_node_vars, input_node_params,
                  mars, data, example_ids, Int32(1), Int32(1))
    kernel = @cuda name="input_flows_circuit" launch=false input_flows_circuit_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_example_threads, node_work = 
        balance_threads(num_input_nodes, num_examples, config; mine, maxe)

    args = (flows, bpc.nodes, inparams_aggr, input_node_idxs, input_node_vars, input_node_params,
            mars, data, example_ids, Int32(num_example_threads), Int32(node_work))
    if debug
        println("Flows of input nodes")
        @show threads blocks num_example_threads node_work num_nodes num_examples
        CUDA.@time kernel(args...; threads, blocks)
    else
        kernel(args...; threads, blocks)
    end

    nothing
end

##################################################################################
# Full downward pass
##################################################################################

function probs_flows_circuit(flows, mars, edge_aggr, inparams_aggr, bpc, data, example_ids; mine, maxe, debug=false)
    eval_circuit(mars, bpc, data, example_ids; mine, maxe, debug)
    flows_circuit(flows, edge_aggr, bpc, mars, length(example_ids); mine, maxe, debug)
    input_flows_circuit(flows, inparams_aggr, bpc, mars, data, example_ids; mine, maxe, debug)
    nothing
end