
"""
    sample(bpc::CuBitsProbCircuit, num_samples::Int, num_rand_vars::Int, types; rng=default_rng())

Generate `num_samples` from the joint distribution of the circuit without any conditions.
Samples are genearted on the GPU. 

 - `bpc`: Circuit on gpu (CuBitProbCircuit)
 - `num_samples`: how many samples to generate
 - `num_rand_vars`: number of random variables in the circuit
 - `types`: Array of possible input types
 - `rng`: (Optional) Random Number Generator


The size of returned Array is `(num_samples, 1, size(data, 2))`.
"""
function sample(bpc::CuBitsProbCircuit, num_samples::Int, num_rand_vars::Int, types; 
        rng = default_rng(), mars_mem=nothing, mine=2, maxe=32, debug=false)
    data = CuMatrix{Union{Missing, types...}}([missing for j=1:1, i=1:num_rand_vars])
    sample(bpc, num_samples, data; rng, debug)
end

"""
    sample(bpc::CuBitsProbCircuit, num_samples, data::CuMatrix; rng=default_rng())

Generate `num_samples` for each datapoint in `data` from the joint distribution of the circuit conditioned on the `data`.
Samples are generated using GPU. 

 - `bpc`: Circuit on gpu (CuBitProbCircuit)
 - `num_samples`: how many samples to generate
 - `rng`: (Optional) Random Number Generator

The size of returned CuArray is `(num_samples, size(data, 1), size(data, 2))`.
"""
function sample(bpc::CuBitsProbCircuit, num_samples, data::CuMatrix;  
    mars_mem=nothing, mine=2, maxe=32,
    rng=default_rng(), debug=false)

    @assert num_samples > 0

    num_examples = size(data, 1)
    num_nodes = length(bpc.nodes)
    
    states = CuArray{Union{Missing,eltype(data)}}(undef, num_samples, num_examples, size(data, 2))
    
    # for now only do all of marginals in one batch
    batch = 1:num_examples
    marginals = prep_memory(mars_mem, (num_examples, num_nodes), (false, true))
    eval_circuit(marginals, bpc, data, batch; mine, maxe, debug)
    sample_downward!(marginals, bpc, data, states, batch, rng; debug)
    cleanup_memory(marginals, mars_mem)
    return states
end

struct CuStack2D
    # parallel grid of stacks
    # size = (num_samples, num_examples, num_features + 3)
    mem::CuArray{Int32, 3} 

    # Index of Top of each stack for each example
    tops::CuMatrix{UInt32}

    CuStack2D(samples, examples, features) = begin
        new(CUDA.zeros(Int32, samples, examples, features + 3), 
            CUDA.zeros(UInt32, samples, examples))
    end
end

function pop_cuda!(stack_mem, stack_tops, i, j)
    # Empty Stack
    if stack_tops[i, j] == zero(UInt32)
        return zero(UInt32) 
    else
        val = stack_mem[i, j, stack_tops[i, j]]
        stack_tops[i, j] -= one(UInt32)
        return val
    end
end

function push_cuda!(stack_mem, stack_tops, val, i, j)
    stack_tops[i, j] += one(eltype(stack_mem))
    CUDA.@cuassert stack_tops[i, j] <= size(stack_mem, 3) "CUDA stack overflow"
    stack_mem[i, j, stack_tops[i, j]] = val
    return nothing
end

function all_empty(stack_tops)
    all(x -> iszero(x), stack_tops)
end

function balance_threads_2d(num_examples, num_decisions, config)
    total_threads_per_block = config.threads
    lsb(n) = n ⊻ ( n& (n-1))
    ratio_diff(a, b) =  ceil(Int, num_examples/a) - ceil(Int, num_decisions/b)

    n_lsb = lsb(total_threads_per_block)
    options_d1 = [Int32(2^i) for i = 0 : log2(n_lsb)]
    append!(options_d1, [total_threads_per_block / n_lsb * Int32(2^i) for i = 0 : log2(n_lsb)])
    options_d2 = [Int32(total_threads_per_block / d1) for d1 in options_d1]

    best_d1 = options_d1[1]
    best_d2 = options_d2[1]
    best_ratio = ratio_diff(best_d1, best_d2)

    for (d1, d2) in zip(options_d1, options_d2)
        cur_ratio = ratio_diff(d1, d2)
        if abs(best_ratio) > abs(cur_ratio)
            best_d1 = d1
            best_d2 = d2
            best_ratio = cur_ratio
        end
    end
    threads = (best_d1, best_d2)    
    blocks = (ceil(Int, num_examples / threads[1]), 
                ceil(Int, num_decisions / threads[2]))

    threads, blocks
end

function sample_downward!(marginals, bpc, data, states, batch, rng; debug)
    CUDA.seed!(rand(rng, UInt))

    num_examples = length(batch)
    num_samples = size(states, 1)
    num_nodes = length(bpc.nodes)
    
    stack = CuStack2D(num_samples, num_examples, size(states, 3))
    # Push root node to all stacks
    stack.tops .= 1
    stack.mem[:,:, 1] .= num_nodes

    CUDA.@sync while true
        rands = CUDA.rand(num_samples, num_examples)
        dummy_args = (marginals, data, states, stack.mem, stack.tops, 
                        bpc.nodes, bpc.node_begin_end, bpc.edge_layers_up.vectors, 
                        bpc.heap, batch, rands)

        kernel = @cuda name="sample_downward!" launch=false sample_downward_kernel!(dummy_args...)
        config = launch_configuration(kernel.fun)    
        threads, blocks = balance_threads_2d(num_samples, num_examples, config)
        args = (marginals, data, states, stack.mem, stack.tops, 
                bpc.nodes, bpc.node_begin_end, bpc.edge_layers_up.vectors, 
                bpc.heap, batch, rands)

        if debug
            print("sample downward step")
            CUDA.@time kernel(args...; threads, blocks)
        else
            kernel(args...; threads, blocks)
        end
        all_empty(stack.tops) && break
    end
    return nothing
end

function sample_downward_kernel!(marginals, data, states, stack_mem, stack_tops, 
    nodes, node_begin_end, edges, 
    heap, batch, rands)

    index_x = ((blockIdx().x - 1) * blockDim().x + threadIdx().x)
    index_y = ((blockIdx().y - 1) * blockDim().y + threadIdx().y)
    stride_x = (blockDim().x * gridDim().x)
    stride_y = (blockDim().y * gridDim().y)
    for s_id = index_x:stride_x:size(states,1)
        for ex_id = index_y:stride_y:size(batch, 1)
            cur_node_id = pop_cuda!(stack_mem, stack_tops, s_id, ex_id)
            if cur_node_id > zero(eltype(stack_mem))
                cur_node = nodes[cur_node_id]
                if cur_node isa BitsInput
                    #### sample the input if missing
                    example_id = batch[ex_id]
                    if ismissing(data[example_id, cur_node.variable])
                        # marginals[ex_id, this_node] should be log(1) = 0 (because missing), so don't need to add that
                        threshold = CUDA.log(rands[s_id, ex_id])
                        sample_value = sample_state(dist(cur_node), threshold, heap)
                        states[s_id, example_id, cur_node.variable] = sample_value
                    else
                        states[s_id, example_id, cur_node.variable] = data[example_id, cur_node.variable]
                    end
                elseif cur_node isa BitsSum
                    #### Choose which child of sum node to sample
                    chosen_edge = node_begin_end[cur_node_id].second ## give all numerical error probability to the last node
                    cumul_prob = typemin(Float32)
                    parent_node_id = edges[node_begin_end[cur_node_id].first].parent_id
                    threshold = CUDA.log(rands[s_id, ex_id]) + marginals[ex_id, parent_node_id]
                    for edge_ind = node_begin_end[cur_node_id].first: node_begin_end[cur_node_id].second
                        edge = edges[edge_ind] 

                        child_prob = marginals[ex_id, edge.prime_id]
                        if edge.sub_id != zero(UInt32)
                            child_prob += marginals[ex_id, edge.sub_id]
                        end
                        if edge isa SumEdge
                            child_prob += edge.logp
                        end
                        cumul_prob = logsumexp(cumul_prob, child_prob)
                        if cumul_prob > threshold
                            chosen_edge = edge_ind
                            break
                        end
                    end

                    # Push the chosen edge into stack 
                    cur_edge = edges[chosen_edge]
                    push_cuda!(stack_mem, stack_tops, cur_edge.prime_id, s_id, ex_id)
                    if cur_edge.sub_id != zero(UInt32)
                        push_cuda!(stack_mem, stack_tops, cur_edge.sub_id, s_id, ex_id)
                    end
                elseif cur_node isa BitsMul
                    #### Just Push all children to stack
                    for edge_ind = node_begin_end[cur_node_id].first: node_begin_end[cur_node_id].second
                        edge = edges[edge_ind]
                        push_cuda!(stack_mem, stack_tops, edge.prime_id, s_id, ex_id)
                        if edge.sub_id != zero(UInt32)
                            push_cuda!(stack_mem, stack_tops, edge.sub_id, s_id, ex_id)
                        end
                    end
                end
            end
        end
    end
    nothing
end
