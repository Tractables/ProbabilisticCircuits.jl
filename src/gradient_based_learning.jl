export sgd_parameter_learning

using LogicCircuits: num_nodes
using StatsFuns: logaddexp
using CUDA

function sgd_parameter_learning(pc::ProbCircuit, data; lr::Float64 = 0.01, 
                                use_sample_weights::Bool = true, use_gpu::Bool = isgpu(data),
                                reuse_values = nothing, reuse_flows = nothing,
                                reuse = (nothing, nothing))
    # Construct the low-level circuit representation
    pbc = ParamBitCircuit(pc, data)
    if use_gpu
        pbc = to_gpu(pbc)
    end
    
    # Main training loop
    @assert isbatched(data)
    for idx = 1 : length(data)
        sgd_parameter_learning(pbc, data[idx]; lr, use_sample_weights, use_gpu,
                               reuse_values, reuse_flows, reuse)
    end
    
    # Store the updated parameters back to `pc`
    estimate_parameters_cached!(pc, pbc)
    
    pbc.params
end
function sgd_parameter_learning(pbc::ParamBitCircuit, data; lr::Float64 = 0.01, 
                                use_sample_weights::Bool = true, use_gpu::Bool = isgpu(data),
                                reuse_values = nothing, reuse_flows = nothing,
                                reuse = (nothing, nothing))
    # Extract weights from dataset
    if isweighted(data)
        # `data' is weighted according to its `weight' column
        data, weights = split_sample_weights(data)
    else
        use_sample_weights = false
    end
    
    # Move data to GPU if needed
    if isgpu(data)
        use_gpu = true
    elseif use_gpu && !isgpu(data)
        data = to_gpu(data)
    end
    
    if use_gpu && !isgpu(pbc)
        pbc.bitcircuit = to_gpu(pbc.bitcircuit)
        pbc.params = to_gpu(pbc.params)
    end
    
    if use_gpu
        if !isgpu(pbc)
            pbc.bitcircuit = to_gpu(pbc.bitcircuit)
            pbc.params = to_gpu(pbc.params)
        end

        if use_sample_weights
            if !isgpu(weights)
                weights = to_gpu(weights)
            end
            sgd_parameter_learning_gpu(pbc, data; lr, weights, reuse_values, reuse_flows, reuse)
        else
            sgd_parameter_learning_gpu(pbc, data; lr, reuse_values, reuse_flows, reuse)
        end
    else
        if use_sample_weights
            sgd_parameter_learning_cpu(pbc, data; lr, weights, reuse_values, reuse_flows, reuse)
        else
            sgd_parameter_learning_cpu(pbc, data; lr, reuse_values, reuse_flows, reuse)
        end
    end
    
    pbc.params
end

function sgd_parameter_learning_cpu(pbc::ParamBitCircuit, data; lr::Float64 = 0.01, 
                                    weights = nothing, reuse_values = nothing, reuse_flows = nothing,
                                    reuse = (nothing, nothing))
    bc = pbc.bitcircuit
    params = pbc.params
    
    # Forward/upward pass
    values = marginal_all(pbc, data, reuse_values)
    
    # Compute the gradient (this part can be customized)
    log_grads = similar!(reuse[1], typeof(params), num_examples(data))
    @inbounds @views log_grads[:] .= zero(eltype(values)) .- values[:, num_nodes(pbc.bitcircuit)]
    
    param_grads = similar!(reuse[2], typeof(params), size(params)...)
    
    # Backward/downward pass
    flows = backprop_flows_down_cpu(pbc, log_grads, values, param_grads, reuse_flows; weights)
    
    # Apply gradients
    apply_gradients_cpu(pbc, param_grads; lr)
    
    pbc.params, values, flows, (log_grads, param_grads)
end

function sgd_parameter_learning_gpu(pbc::ParamBitCircuit, data; lr::Float64 = 0.01, 
                                    weights = nothing, reuse_values = nothing, reuse_flows = nothing,
                                    reuse = (nothing, nothing))
    bc = pbc.bitcircuit
    params = pbc.params
    
    # Forward/upward pass
    values = marginal_all(pbc, data, reuse_values)
    
    # Compute the gradient (this part can be customized)
    log_grads = similar!(reuse[1], typeof(params), num_examples(data))
    @inbounds @views log_grads[:] .= zero(eltype(values)) .- values[:, num_nodes(pbc.bitcircuit)]
    
    param_grads = similar!(reuse[2], typeof(params), size(params)...)
    
    # Backward/downward pass
    flows = backprop_flows_down_gpu(pbc, log_grads, values, param_grads, reuse_flows; weights)
    
    # Apply gradients
    apply_gradients_gpu(pbc, param_grads; lr)
    
    pbc.params, values, flows, (log_grads, param_grads)
end

######################################
# Bit circuit backprop downward pass #
######################################

function backprop_flows_down_cpu(pbc::ParamBitCircuit, log_grads, values, param_grads, reuse = nothing; weights = nothing)
    flows = similar!(reuse, typeof(values), size(values)...)
    
    if weights !== nothing
        @inbounds @views log_grads .+= log.(weights)
    end
    
    backprop_flows_down_layers_cpu(pbc, log_grads, flows, values, param_grads)
    
    flows # Return for reuse
end

function backprop_flows_down_layers_cpu(pbc::ParamBitCircuit, log_grads::Vector, flows::Matrix, values::Matrix,
                                    param_grads::Vector)
    @assert flows !== values
    bc::BitCircuit = pbc.bitcircuit
    els = bc.elements
    for layer in Iterators.reverse(bc.layers)
        ## Compute gradients
        Threads.@threads for dec_id in layer
            par_start = @inbounds bc.nodes[3, dec_id]
            if iszero(par_start)
                if dec_id == num_nodes(bc)
                    # Assign (log) gradient to the root node
                    @inbounds @views flows[:, dec_id] .= log_grads
                end
                # no parents, ignore (can happen for false/true node and root)
            else
                par_end = @inbounds bc.nodes[4, dec_id]
                for j = par_start : par_end
                    par = @inbounds bc.parents[j] # Parent #j of `dec_id`
                    grandpa = @inbounds els[1, par] # Parent of `par`
                    sib_id = sibling(els, par, dec_id) # Sibling of `dec_id` w.r.t. `par`
                    
                    single_child = has_single_child(bc.nodes, grandpa)
                    if single_child
                        if j == par_start
                            @inbounds @views @. flows[:, dec_id] = flows[:, grandpa] + values[:, sib_id]
                        else
                            @inbounds @simd for i = 1 : size(flows, 1)
                                flows[i, dec_id] = logaddexp(flows[i, dec_id], flows[i, grandpa] + values[i, sib_id])
                            end
                        end
                    else
                        θ = eltype(flows)(pbc.params[par])
                        if j == par_start
                            @inbounds @views @. flows[:, dec_id] = flows[:, grandpa] + values[:, sib_id] + θ
                        else
                            @inbounds @simd for i = 1 : size(flows, 1)
                                flows[i, dec_id] = logaddexp(flows[i, dec_id], flows[i, grandpa] + values[i, sib_id] + θ)
                            end
                        end
                    end
                    # Compute gradient only once
                    if sib_id > dec_id
                        param_grads[par] = sum(exp.(flows[:, grandpa] + values[:, dec_id] + values[:, sib_id]))
                    end
                end
            end
        end
    end
end

function apply_gradients_cpu(pbc::ParamBitCircuit, param_grads::Vector; lr::Float64 = 0.01)
    bc::BitCircuit = pbc.bitcircuit
    nodes = bc.nodes
    params = pbc.params
    for layer in Iterators.reverse(bc.layers)
        ## Apply gradients
        Threads.@threads for dec_id in layer
            if !has_single_child(nodes, dec_id)
                ele_start_id = nodes[1, dec_id]
                ele_end_id = nodes[2, dec_id]
                
                sum_grads = 0.0
                @inbounds for ele_id = ele_start_id : ele_end_id
                    sum_grads += param_grads[ele_id]
                end
                sum_params = -Inf
                @inbounds for ele_id = ele_start_id : ele_end_id
                    params[ele_id] += lr * param_grads[ele_id] / (sum_grads + 1e-8)
                    sum_params = logaddexp(sum_params, params[ele_id])
                end
                @inbounds for ele_id = ele_start_id : ele_end_id
                    params[ele_id] -= sum_params
                end
            end
        end
    end
end

function backprop_flows_down_gpu(pbc::ParamBitCircuit, log_grads::CuVector, values::CuMatrix,
                                 param_grads::CuVector, reuse =  nothing; dec_per_thread = 8, 
                                 log2_threads_per_block = 7, weights = nothing)
    flows = similar!(reuse, typeof(values), size(values)...)
    if weights !== nothing
        @inbounds @views log_grads .+= CUDA.log.(weights)
    end
    
    bc::BitCircuit = pbc.bitcircuit
    @inbounds @views param_grads .= zero(eltype(param_grads))
    param_grads_device = CUDA.cudaconvert(param_grads)
    
    CUDA.@sync for layer in Iterators.reverse(bc.layers)
        num_examples = size(values, 1)
        num_decision_sets = length(layer)/dec_per_thread
        num_threads =  balance_threads(num_examples, num_decision_sets, log2_threads_per_block)
        num_blocks = (ceil(Int, num_examples/num_threads[1]), 
                      ceil(Int, num_decision_sets/num_threads[2])) 
        @cuda threads=num_threads blocks=num_blocks backprop_flows_down_layers_cuda(layer, bc.nodes, bc.elements, bc.parents, pbc.params, log_grads, flows, values, param_grads_device)
    end
    
    flows # Return for reuse
end

function backprop_flows_down_layers_cuda(layer, nodes, elements, parents, params, log_grads, flows, values, param_grads)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().y
    
    for k = index_x : stride_x : size(values, 1)
        for i = index_y : stride_y : length(layer)
            dec_id = @inbounds layer[i]
            if dec_id == size(nodes, 2)
                # Assign (log) gradient to the root node
                flow = log_grads[k]
            else
                par_start = @inbounds nodes[3, dec_id]
                flow = typemin(eltype(flows)) # log(0)
                if !iszero(par_start)
                    par_end = @inbounds nodes[4, dec_id]
                    for j = par_start : par_end
                        par = @inbounds parents[j]
                        grandpa = @inbounds elements[1, par]
                        sib_id = sibling(elements, par, dec_id)
                        
                        g_flow = @inbounds flows[k, grandpa]
                        d_value = @inbounds values[k, dec_id]
                        s_value = @inbounds values[k, sib_id]
                        if has_single_child(nodes, grandpa)
                            edge_flow = g_flow + s_value
                        else
                            θ = eltype(flows)(params[par])
                            edge_flow = g_flow + s_value + θ
                        end
                        flow = logsumexp_cuda(flow, edge_flow)
                        
                        # Compute gradient only once
                        if sib_id > dec_id
                            grad::Float64 = CUDA.exp(g_flow + d_value + s_value)
                            CUDA.@atomic param_grads[par] += grad
                        end
                    end
                end
            end
            @inbounds flows[k, dec_id] = flow
        end
    end
    return nothing
end

function apply_gradients_gpu(pbc::ParamBitCircuit, param_grads::CuVector; lr::Float64 = 0.01)
    bc::BitCircuit = pbc.bitcircuit
    
    CUDA.@sync for layer in Iterators.reverse(bc.layers)
        num_threads = 2^min(ceil(Int, 2.0 * log2(length(layer))), 8)
        num_blocks = 2^ceil(Int, log2(length(layer)^2 / num_threads))
        @cuda threads=num_threads blocks=num_blocks apply_gradients_cuda(layer, bc.nodes, param_grads, pbc.params, 
                                                                         lr::Float64)
    end
end

function apply_gradients_cuda(layer, nodes, param_grads, params, lr::Float64)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride_x = blockDim().x * gridDim().x
    
    for i = index_x : stride_x : length(layer)
        dec_id = @inbounds layer[i]
        single_child = has_single_child(nodes, dec_id)
        if !single_child
            ele_start_id = nodes[1, dec_id]
            ele_end_id = nodes[2, dec_id]

            sum_grads = zero(eltype(param_grads))
            @inbounds for ele_id = ele_start_id : ele_end_id
                sum_grads += param_grads[ele_id]
            end
            sum_params = -Inf
            @inbounds for ele_id = ele_start_id : ele_end_id
                params[ele_id] += lr * param_grads[ele_id] / (sum_grads + 1e-8)
                sum_params = logsumexp_cuda(sum_params, params[ele_id])
            end
            @inbounds for ele_id = ele_start_id : ele_end_id
                params[ele_id] -= sum_params
            end
        end
    end
end