using CUDA, Random

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
# Pseudocounts for inputs
##################################################################################

function add_pseudocount_input_kernel(inparams_aggr, nodes, input_node_idxs, pseudocount, 
                                      num_node_threads::Int32, param_work::Int32)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, node_id = fldmod1(threadid, num_node_threads)

    ninput_nodes = length(input_node_idxs)
    ninput_params = size(inparams_aggr, 2)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), ninput_params)

    @inbounds if node_id <= ninput_nodes
        for param_id = param_start : param_end
            orig_node_id::UInt32 = input_node_idxs[node_id]
            node::BitsInputNode = nodes[orig_node_id]
            glob_id = node.global_id::UInt32
            dist_type = leaf.dist_type::UInt8

            if dist_type == UInt8(1) # LiteralDist
                nothing # need to do nothing
            elseif dist_type == UInt8(2) # BernoulliDist
                if param_id <= 2
                    inparams_aggr[glob_id, param_id] += pseudocount * Float32(0.5)
                end
            elseif dist_type == UInt8(3) # CategoricalDist
                nparams = node.num_params::UInt32
                if param_id <= nparams
                    inparams_aggr[glob_id, param_id] += pseudocount / nparams
                end
            else
                @assert false
            end
        end
    end

    nothing
end

function add_pseudocount_input(inparams_aggr, bpc, pseudocount; mine=2, maxe=32)
    nodes = bpc.nodes
    input_node_idxs = bpc.input_node_idxs

    dummy_args = (inparams_aggr, nodes, input_node_idxs, Float32(pseudocount), Int32(1), Int32(1))
    kernel = @cuda name="add_pseudocount_input" launch=false add_pseudocount_input_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_node_threads, param_work = 
        balance_threads(size(inparams_aggr, 2), size(inparams_aggr, 1), config; mine, maxe)

    args = (inparams_aggr, nodes, input_node_idxs, Float32(pseudocount), Int32(num_node_threads), Int32(param_work))
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
# Aggregate node flows for inputs
##################################################################################

function aggr_node_flows_input_kernel(inparams_aggr, innode_aggr, nodes, input_node_idxs, pseudocount, 
                                      num_node_threads::Int32, param_work::Int32)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, node_id = fldmod1(threadid, num_node_threads)

    ninput_nodes = length(input_node_idxs)
    ninput_params = size(inparams_aggr, 2)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), ninput_params)

    @inbounds if node_id <= ninput_nodes
        for param_id = param_start : param_end
            orig_node_id::UInt32 = input_node_idxs[node_id]
            node::BitsInputNode = nodes[orig_node_id]
            glob_id = node.global_id::UInt32
            dist_type = leaf.dist_type::UInt8

            if dist_type == UInt8(1) # LiteralDist
                nothing # need to do nothing
            elseif dist_type == UInt8(2) # BernoulliDist
                if param_id <= 2
                    CUDA.@atomic innode_aggr[glob_id] += inparams_aggr[glob_id, param_id]
                end
            elseif dist_type == UInt8(3) # CategoricalDist
                nparams = node.num_params::UInt32
                if param_id <= nparams
                    CUDA.@atomic innode_aggr[glob_id] += inparams_aggr[glob_id, param_id]
                end
            else
                @assert false
            end
        end
    end

    nothing
end

function aggr_node_flows_input(inparams_aggr, innode_aggr, bpc; mine=2, maxe=32)
    nodes = bpc.nodes
    input_node_idxs = bpc.input_node_idxs

    dummy_args = (inparams_aggr, innode_aggr, nodes, input_node_idxs, Int32(1), Int32(1))
    kernel = @cuda name="aggr_node_flows_input" launch=false aggr_node_flows_input_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_node_threads, param_work = 
        balance_threads(size(inparams_aggr, 2), size(inparams_aggr, 1), config; mine, maxe)

    args = (inparams_aggr, innode_aggr, nodes, input_node_idxs, Int32(num_node_threads), Int32(param_work))
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

##################################################################################
# Update parameters for inputs
##################################################################################

function update_params_input_kernel(inparams_aggr, innode_aggr, nodes, input_node_idxs, input_node_params, inertia::Float32,
                                    num_node_threads::Int32, param_work::Int32)

    threadid = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x

    param_batch, node_id = fldmod1(threadid, num_node_threads)

    ninput_nodes = length(input_node_idxs)
    ninput_params = size(inparams_aggr, 2)

    param_start = one(Int32) + (param_batch - one(Int32)) * param_work
    param_end = min(param_start + param_work - one(Int32), ninput_params)

    @inbounds if node_id <= ninput_nodes
        for param_id = param_start : param_end
            orig_node_id::UInt32 = input_node_idxs[node_id]
            node::BitsInputNode = nodes[orig_node_id]
            glob_id = node.global_id::UInt32
            dist_type = leaf.dist_type::UInt8

            if dist_type == UInt8(1) # LiteralDist
                nothing # need to do nothing

            elseif dist_type == UInt8(2) # BernoulliDist
                if param_id == 1
                    old = inertia * exp(input_node_params[glob_id, 1])
                    new = (one(Float32) - inertia) * inparams_aggr[glob_id, 1] / innode_aggr[glob_id]
                    new_log_param = log(old + new)
                    input_node_params[glob_id, 1] = new_log_param
                end

            elseif dist_type == UInt8(3) # CategoricalDist
                nparams = node.num_params::UInt32
                if param_id <= nparams
                    old = inertia * exp(input_node_params[glob_id, param_id])
                    new = (one(Float32) - inertia) * inparams_aggr[glob_id, param_id] / innode_aggr[glob_id]
                    new_log_param = log(old + new)
                    input_node_params[glob_id, param_id] = new_log_param
                end

            else
                @assert false
            end
        end
    end

    nothing
end

function update_params_input(bpc, innode_aggr, inparams_aggr; inertia = 0, mine = 2, maxe = 32)
    nodes = bpc.nodes
    input_node_idxs = bpc.input_node_idxs
    input_node_params = bpc.input_node_params

    dummy_args = (inparams_aggr, innode_aggr, nodes, input_node_idxs, input_node_params, Float32(inertia), Int32(1), Int32(1))
    kernel = @cuda name="update_params_input" launch=false update_params_input_kernel(dummy_args...)
    config = launch_configuration(kernel.fun)

    threads, blocks, num_node_threads, param_work = 
        balance_threads(size(input_node_params, 2), size(input_node_params, 1), config; mine, maxe)

    args = (inparams_aggr, innode_aggr, nodes, input_node_idxs, input_node_params, Float32(inertia), Int32(num_node_threads), Int32(param_work))
    kernel(args...; threads, blocks)

    nothing
end

#######################
### Full-Batch EM
######################

"Turn binary data into floating point data close to 0 and 1."
function soften_data(data; softness, pseudocount=1)
    data_marginals = ((sum(data; dims=1) .+ Float32(pseudocount/2)) 
                        ./ Float32(size(data, 1) + pseudocount))
    Float32(1-softness) * data .+ Float32(softness) * data_marginals
end

function full_batch_em_step(bpc::CuBitsProbCircuit, data::CuArray; 
                            batch_size, pseudocount, report_ll=true,
                            marginals, flows, node_aggr, edge_aggr, 
                            inparams_aggr, innode_aggr,
                            mine, maxe, debug)

    num_examples = size(data)[1]
    num_batches = cld(num_examples, batch_size)
    
    if report_ll 
        log_likelihoods = CUDA.zeros(Float32, num_batches, 1)
    end
    
    edge_aggr .= zero(Float32)
    inparams_aggr .= zero(Float32)
    batch_index = 0

    for batch_start = 1:batch_size:num_examples

        batch_end = min(batch_start+batch_size-1, num_examples)
        batch = batch_start:batch_end
        num_batch_examples = batch_end - batch_start + 1
        batch_index += 1

        probs_flows_circuit(flows, marginals, edge_aggr, inparams_aggr, bpc, data, batch; 
                            mine, maxe, debug)

        if report_ll
            @views sum!(
                log_likelihoods[batch_index:batch_index, 1:1], 
                marginals[1:num_batch_examples,end:end])
        end
    end

    add_pseudocount(edge_aggr, node_aggr, bpc, pseudocount)
    aggr_node_flows(node_aggr, bpc, edge_aggr)
    update_params(bpc, node_aggr, edge_aggr; inertia = 0)

    add_pseudocount_input(inparams_aggr, bpc, pseudocount)
    aggr_node_flows_input(inparams_aggr, innode_aggr, bpc)
    update_params_input(bpc, innode_aggr, inparams_aggr; inertia = 0)

    return report_ll ? sum(log_likelihoods) / num_examples : 0.0
end

function full_batch_em(bpc::CuBitsProbCircuit, raw_data::CuArray, num_epochs; 
                       batch_size, pseudocount, softness, report_ll=true,
                       mars_mem = nothing, flows_mem = nothing, node_aggr_mem = nothing, 
                       edge_aggr_mem = nothing, inparams_aggr_mem = nothing, innode_aggr_mem = nothing, 
                       mine=2, maxe=32, debug = false)
    
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    data = iszero(softness) ? raw_data : soften_data(raw_data; softness)

    marginals = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    node_aggr = prep_memory(node_aggr_mem, (num_nodes,))
    edge_aggr = prep_memory(edge_aggr_mem, (num_edges,))
    inparams_aggr = prep_memory(inparams_aggr_mem, (length(bpc.input_node_idxs), bpc.inparams_aggr_size), (true, true))
    innode_aggr = prep_memory(innode_aggr_mem, (length(bpc.input_node_idxs),))

    for epoch = 1:num_epochs
        log_likelihood = full_batch_em_step(bpc, data; 
            batch_size, pseudocount, report_ll,
            marginals, flows, node_aggr, edge_aggr, inparams_aggr, innode_aggr,
            mine, maxe, debug)
        println("Full-batch EM epoch $epoch; train LL $log_likelihood")
    end
    
    cleanup_memory((data, raw_data), (flows, flows_mem), 
        (node_aggr, node_aggr_mem), (edge_aggr, edge_aggr_mem), 
        (inparams_aggr, inparams_aggr_mem), (innode_aggr, innode_aggr_mem))

    log_likelihood
end

#######################
### Mini-Batch EM
######################

function mini_batch_em(bpc::CuBitsProbCircuit, raw_data::CuArray, num_epochs; 
                       batch_size, pseudocount, softness, 
                       param_inertia, param_inertia_end = param_inertia, 
                       flow_memory = 0, flow_memory_end = flow_memory, 
                       shuffle=:each_epoch,  
                       mars_mem = nothing, flows_mem = nothing, node_aggr_mem = nothing, edge_aggr_mem = nothing,
                       inparams_aggr_mem = nothing, innode_aggr_mem = nothing,
                       mine=2, maxe=32, debug=false)

    @assert pseudocount >= 0
    @assert 0 <= param_inertia <= 1
    @assert param_inertia <= param_inertia_end <= 1
    @assert 0 <= flow_memory  
    @assert flow_memory <= flow_memory_end  
    @assert shuffle ∈ [:never, :once, :each_epoch, :each_batch]

    num_examples = size(raw_data)[1]
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    num_batches = num_examples ÷ batch_size # drop last incomplete batch

    @assert batch_size <= num_examples
    
    data = iszero(softness) ? raw_data : soften_data(raw_data; softness)

    marginals = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    node_aggr = prep_memory(node_aggr_mem, (num_nodes,))
    edge_aggr = prep_memory(edge_aggr_mem, (num_edges,))
    inparams_aggr = prep_memory(inparams_aggr_mem, (length(bpc.input_node_idxs), bpc.inparams_aggr_size), (true, true))
    innode_aggr = prep_memory(innode_aggr_mem, (length(bpc.input_node_idxs),))

    edge_aggr .= zero(Float32)
    inparams_aggr .= zero(Float32)
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

    Δparam_inertia = (param_inertia_end-param_inertia)/num_epochs
    Δflow_memory = (flow_memory_end-flow_memory)/num_epochs

    for epoch in 1:num_epochs

        log_likelihood = zero(Float32)

        (shuffle == :each_epoch) && do_shuffle()

        for batch in batches

            (shuffle == :each_batch) && do_shuffle()

            if iszero(flow_memory)
                edge_aggr .= zero(Float32)
                inparams_aggr .= zeros(Float32)
            else
                # slowly forget old edge aggregates
                rate = max(zero(Float32), one(Float32) - (batch_size + pseudocount) / flow_memory)
                edge_aggr .*= rate
                inparams_aggr .*= rate
            end

            probs_flows_circuit(flows, marginals, edge_aggr, inparams_aggr, bpc, data, batch; 
                                mine, maxe, debug)
            
            log_likelihood += sum(output_layer) / batch_size

            add_pseudocount(edge_aggr, node_aggr, bpc, pseudocount)
            aggr_node_flows(node_aggr, bpc, edge_aggr)
            update_params(bpc, node_aggr, edge_aggr; inertia = param_inertia)

            add_pseudocount_input(inparams_aggr, bpc, pseudocount)
            aggr_node_flows_input(inparams_aggr, innode_aggr, bpc)
            update_params_input(bpc, innode_aggr, inparams_aggr; inertia = param_inertia)
            
        end
            
        log_likelihood /= num_batches
        println("Mini-batch EM iter $epoch; train LL $log_likelihood")

        param_inertia += Δparam_inertia
        flow_memory += Δflow_memory
    end

    cleanup_memory((data, raw_data), (flows, flows_mem), 
        (node_aggr, node_aggr_mem), (edge_aggr, edge_aggr_mem), 
        (inparams_aggr, inparams_aggr_mem), (innode_aggr, innode_aggr_mem))
    CUDA.unsafe_free!(shuffled_indices)

    nothing
end
