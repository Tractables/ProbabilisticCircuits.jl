using CUDA, Random

export full_batch_em, mini_batch_em, init_parameters

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

##################################################################################
# Clear memory of input nodes
##################################################################################

function clear_input_node_mem_kernel(nodes, input_node_ids, heap, rate)

    node_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if node_id <= length(input_node_ids)
        orig_node_id::UInt32 = input_node_ids[node_id]
        inputnode = nodes[orig_node_id]::BitsInput
        clear_memory(dist(inputnode), heap, rate)
    end
    nothing
end

function clear_input_node_mem(bpc; rate = 0)
    num_input_nodes = length(bpc.input_node_ids)

    args = (bpc.nodes, bpc.input_node_ids, bpc.heap, Float32(rate))
    kernel = @cuda name="clear_input_node_mem" launch=false clear_input_node_mem_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(num_input_nodes, threads)

    kernel(args...; threads, blocks)
    nothing
end

##################################################################################
# Update parameters of input nodes
##################################################################################

function update_input_node_params_kernel(nodes, input_node_ids, heap, pseudocount, inertia)

    node_id = ((blockIdx().x - one(Int32)) * blockDim().x) + threadIdx().x 

    @inbounds if node_id <= length(input_node_ids)
        orig_node_id::UInt32 = input_node_ids[node_id]
        inputnode = nodes[orig_node_id]::BitsInput
        update_params(dist(inputnode), heap, pseudocount, inertia)
    end
    nothing
end

function update_input_node_params(bpc; pseudocount, inertia = 0)
    num_input_nodes = length(bpc.input_node_ids)

    args = (bpc.nodes, bpc.input_node_ids, bpc.heap, Float32(pseudocount), Float32(inertia))
    kernel = @cuda name="update_input_node_params" launch=false update_input_node_params_kernel(args...)
    threads = launch_configuration(kernel.fun).threads
    blocks = cld(num_input_nodes, threads)

    kernel(args...; threads, blocks)
    nothing
end

####################################
### Parameter initialization
####################################

"Initialize parameters of ProbCircuit."
function init_parameters(pc::ProbCircuit; perturbation = 0.0)
    perturbation = Float32(perturbation)
    foreach(pc) do pn
        @inbounds if issum(pn)
            if num_children(pn) == 1
                pn.params .= zero(Float32)
            else
                if perturbation < 1e-8
                    pn.params .= loguniform(num_children(pn))
                else
                    unnormalized_probs = map(x -> one(Float32) - perturbation + x * Float32(2.0) * perturbation, rand(Float32, num_children(pn)))
                    pn.params .= log.(unnormalized_probs ./ sum(unnormalized_probs))
                end
            end
        elseif isinput(pn)
            init_params(pn, perturbation)
        end
    end
    nothing
end

#######################
### Full-Batch EM
#######################

"Turn binary data into floating point data close to 0 and 1."
function soften_data(data; softness, pseudocount=1)
    data_marginals = ((sum(data; dims=1) .+ Float32(pseudocount/2)) 
                        ./ Float32(size(data, 1) + pseudocount))
    Float32(1-softness) * data .+ Float32(softness) * data_marginals
end

function full_batch_em_step(bpc::CuBitsProbCircuit, data::CuArray; 
                            batch_size, pseudocount, report_ll=true,
                            marginals, flows, node_aggr, edge_aggr,
                            mine, maxe, debug)

    num_examples = size(data)[1]
    num_batches = cld(num_examples, batch_size)
    
    if report_ll 
        log_likelihoods = CUDA.zeros(Float32, num_batches, 1)
    end
    
    edge_aggr .= zero(Float32)
    clear_input_node_mem(bpc; rate = 0)
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
    update_params(bpc, node_aggr, edge_aggr; inertia = 0)

    update_input_node_params(bpc; pseudocount, inertia = 0)

    return report_ll ? sum(log_likelihoods) / num_examples : 0.0
end

function full_batch_em(bpc::CuBitsProbCircuit, raw_data::CuArray, num_epochs; 
                       batch_size, pseudocount, softness = 0, report_ll = true,
                       mars_mem = nothing, flows_mem = nothing, node_aggr_mem = nothing, 
                       edge_aggr_mem = nothing, mine=2, maxe=32, debug = false, verbose = true)
    
    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)
    data = iszero(softness) ? raw_data : soften_data(raw_data; softness)

    marginals = prep_memory(mars_mem, (batch_size, num_nodes), (false, true))
    flows = prep_memory(flows_mem, (batch_size, num_nodes), (false, true))
    node_aggr = prep_memory(node_aggr_mem, (num_nodes,))
    edge_aggr = prep_memory(edge_aggr_mem, (num_edges,))

    log_likelihoods = Vector{Float32}()

    for epoch = 1:num_epochs
        log_likelihood = full_batch_em_step(bpc, data; 
            batch_size, pseudocount, report_ll,
            marginals, flows, node_aggr, edge_aggr, 
            mine, maxe, debug)
        push!(log_likelihoods, log_likelihood)

        if verbose
            println("Full-batch EM epoch $epoch; train LL $log_likelihood")
        end
    end
    
    cleanup_memory((data, raw_data), (flows, flows_mem), 
        (node_aggr, node_aggr_mem), (edge_aggr, edge_aggr_mem))

    log_likelihoods
end

#######################
### Mini-Batch EM
######################

function mini_batch_em(bpc::CuBitsProbCircuit, raw_data::CuArray, num_epochs; 
                       batch_size, pseudocount, 
                       param_inertia, param_inertia_end = param_inertia, 
                       flow_memory = 0, flow_memory_end = flow_memory, 
                       softness = 0, shuffle=:each_epoch,  
                       mars_mem = nothing, flows_mem = nothing, node_aggr_mem = nothing, edge_aggr_mem = nothing,
                       mine = 2, maxe = 32, debug = false, verbose = true)

    @assert pseudocount >= 0
    @assert 0 <= param_inertia <= 1
    @assert param_inertia <= param_inertia_end <= 1
    @assert 0 <= flow_memory  
    @assert flow_memory <= flow_memory_end  
    @assert shuffle ∈ [:once, :each_epoch, :each_batch]

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

    edge_aggr .= zero(Float32)
    clear_input_node_mem(bpc; rate = 0)
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

    log_likelihoods = Vector{Float32}()

    for epoch in 1:num_epochs

        log_likelihood = zero(Float32)

        (shuffle == :each_epoch) && do_shuffle()

        for batch in batches

            (shuffle == :each_batch) && do_shuffle()

            if iszero(flow_memory)
                edge_aggr .= zero(Float32)
                clear_input_node_mem(bpc; rate = 0)
            else
                # slowly forget old edge aggregates
                rate = max(zero(Float32), one(Float32) - (batch_size + pseudocount) / flow_memory)
                edge_aggr .*= rate
                clear_input_node_mem(bpc; rate)
            end

            probs_flows_circuit(flows, marginals, edge_aggr, bpc, data, batch; 
                                mine, maxe, debug)
            
            log_likelihood += sum(output_layer) / batch_size

            add_pseudocount(edge_aggr, node_aggr, bpc, pseudocount)
            aggr_node_flows(node_aggr, bpc, edge_aggr)
            update_params(bpc, node_aggr, edge_aggr; inertia = param_inertia)
            
            update_input_node_params(bpc; pseudocount, inertia = param_inertia)
            
        end
            
        log_likelihood /= num_batches
        push!(log_likelihoods, log_likelihood)
        if verbose
            println("Mini-batch EM iter $epoch; train LL $log_likelihood")
        end

        param_inertia += Δparam_inertia
        flow_memory += Δflow_memory
    end

    cleanup_memory((data, raw_data), (flows, flows_mem), 
        (node_aggr, node_aggr_mem), (edge_aggr, edge_aggr_mem))
    CUDA.unsafe_free!(shuffled_indices)

    log_likelihoods
end
