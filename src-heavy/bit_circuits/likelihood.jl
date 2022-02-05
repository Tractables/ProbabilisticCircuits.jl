using CUDA, Random

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
                    else
                        # data could be Bool, or Float after "data softening"
                        if nonmissingtype(eltype(data)) == Bool
                            if (lit > 0) == v
                                zero(Float32)
                            else
                                -Inf32
                            end
                        elseif nonmissingtype(eltype(data)) <: AbstractFloat
                            if (lit > 0)
                                log(v)
                            else
                                log(one(Float32)-v)
                            end
                        else
                            error("Unknown data type $(nonmissingtype(eltype(data)))")
                        end
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
        # TODO reinstate @fastmath below when https://github.com/JuliaGPU/CUDA.jl/issues/1352 is fixed
        # @fastmath 
        max(x,y) + log1p(exp(-abs(x-y))) 
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

function cleanup_memory(used::CuArray, reused)
    if used !== reused
        CUDA.unsafe_free!(used)
    end
end

function cleanup_memory(used_reused::Tuple...)
    for (used, reused) in used_reused
        cleanup_memory(used, reused)
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

    cleanup_memory(marginals, mars_mem)

    return sum(log_likelihoods) / num_examples
end
