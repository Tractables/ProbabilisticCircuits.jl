

function MAP(bpc::CuBitsProbCircuit, data::CuArray;
    batch_size, mars_mem=nothing,
    mine=2,maxe=32, debug=false)

    num_examples = size(data)[1]
    num_nodes = length(bpc.nodes)
    marginals = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))

    states = deepcopy(data)
    for batch_start = 1:batch_size:num_examples
        batch_end = min(batch_start+batch_size-1, num_examples)
        batch = batch_start:batch_end
        num_batch_examples = length(batch)
        
        eval_circuit_max!(marginals, bpc, data, batch; mine, maxe, debug)
        map_downward!(marginals, bpc, states, batch)
    end
    cleanup_memory(marginals, mars_mem)
    return states
end

struct CuStack
    # parallel stacks for each example (max stack size is features + 3 which is preallocated)
    mem::CuMatrix{Int32} 

    # Index of Top of each stack for each example
    tops::CuArray{UInt32}

    CuStack(examples, features) = begin
        new(CUDA.zeros(Int32, examples, features + 3), 
            CUDA.zeros(UInt32, examples))
    end
end

function pop_cuda!(stack_mem, stack_tops, i)
    # Empty Stack
    if @inbounds stack_tops[i] == zero(UInt32)
        return zero(UInt32) 
    else
        @inbounds local val = stack_mem[i, stack_tops[i]]
        # does it have to be atomic??
        @inbounds stack_tops[i] -= one(UInt32)
        return val
    end
end

function push_cuda!(stack_mem, stack_tops, val, i)
    @inbounds stack_tops[i] += one(eltype(stack))
    @inbounds CUDA.@cuassert stack_tops[i] <= size(stack_mem, 2) "CUDA stack overflow"
    @inbounds stack_mem[i, stack_tops[i]] = val
    return nothing
end


function map_downward!(marginals::CuMatrix, bpc::CuBitsProbCircuit, states, batch)
    stack = CuStack(length(batch), size(states, 2))  
    
    CUDA.@sync begin
        dummy_args = (marginals, states, stack.mem, stack.tops, bpc.nodes, bpc.node_begin_end, bpc.edge_layers_up.vectors, bpc.heap, batch)
        kernel = @cuda name="map_down" launch=false map_downward_kernel!(dummy_args...)
        config = launch_configuration(kernel.fun)    
        threads = config.threads
        blocks = cld(size(state,1), threads)
        kernel(dummy_args... ; threads, blocks)
    end
    nothing
end

function map_downward_kernel!(marginals, states, stack_mem, stack_tops, nodes, node_begin_end, edges, heap, batch)
    index_x = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x
    stride_x = blockDim().x * gridDim().x
    for ex_id = index_x:stride_x:size(state,1)
        cur_node_id = pop_cuda!(stack_mem, stack_tops, ex_id)
        while cur_node_id > zero(UInt32)
            cur_node = nodes[cur_node_id]
            node_type = typeof(nodes[cur_node_id])
            if node_type == BitsInput
                example_id = batch[ex_id]
                if ismissing(states[example_id, node.variable])
                    states[example_id, node.variable] = map_state(node.dist, heap)
                end
            elseif node_type == BitsSum
                max_pr = typemin(Float32)
                chosen_edge = 1
                for edge_ind = node_begin_end[cur_node_id].first: node_begin_end[cur_node_id].second
                    edge = edges[edge_id]
                     # compute max-probability coming from child
                    child_prob = mars[ex_id, edge.prime_id]
                    if edge.sub_id != 0
                        child_prob += mars[ex_id, edge.sub_id]
                    end
                    child_prob += edge.logp
                    
                    if child_prob > max_pr
                        max_pr = child_prob
                        chosen_edge = end_ind
                    end
                end
                # Push the chosen edge into stack 
                cur_edge = edges[chosen_edge]
                push_cuda!(stack_mem, stack_tops, cur_edge.prime_id, ex_id)
                if edge.sub_id != zero(UInt32)
                    push_cuda!(stack_mem, stack_tops, cur_edge.sub_id, ex_id)
                end
            elseif node_type == BitsMul
                for edge_ind = node_begin_end[cur_node_id].first: node_begin_end[cur_node_id].second
                    edge = edges[edge_ind]
                    push_cuda!(stack_mem, stack_tops, edge.prime_id, ex_id)
                    if edge.sub_id != zero(UInt32)
                        push_cuda!(stack_mem, stack_tops, edge.sub_id, ex_id)
                    end
                end
            end
            # Pop the next Node (zero if empty)
            cur_node_id = pop_cuda!(stack_mem, stack_tops, ex_id)
        end
    end
    return nothing
end


# run entire circuit taking mode on inputs and max on sum nodes
function eval_circuit_max!(mars, bpc, data, example_ids; mine, maxe, debug=false)
    input_init_func(dist, heap) = 
        #zero(Float32) 
        map_loglikelihood(dist, heap) # thid does not work for some reason
    
    sum_agg_func(x::Float32, y::Float32) =
        max(x, y)

    init_mar!(mars, bpc, data, example_ids; mine, maxe, input_init_func, debug)
    layer_start = 1
    for layer_end in bpc.edge_layers_up.ends
        layer_up(mars, bpc, layer_start, layer_end, length(example_ids); mine, maxe, sum_agg_func, debug)
        layer_start = layer_end + 1
    end
    nothing
end
