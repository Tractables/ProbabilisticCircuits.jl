using StatsFuns: logsumexp, log1pexp

using CUDA: CUDA, @cuda
using DataFrames: DataFrame
using LoopVectorization: @avx
using LogicCircuits: balance_threads, BitCircuit, balance_threads_2d

export marginal, MAR, marginal_all, marginal_log_likelihood, 
marginal_log_likelihood_avg, marginal_flows, marginal_flows_down

#####################
# Circuit marginal evaluation
#####################
  
"""
Evaluate a probabilistic circuit as a function
"""
function (root::ProbCircuit)(data...)
    marginal(root, data...)
end

"""
    marginal(root::ProbCircuit, data::Union{Real,Missing}...)
    marginal(root::ProbCircuit, data::Union{Vector{Union{Bool,Missing}},CuVector{UInt8}})
    marginal(circuit::ProbCircuit, data::DataFrame)
    marginal(circuit::ParamBitCircuit, data::DataFrame)::AbstractVector
    marginal(circuit::SharedProbCircuit, data::DataFrame, weights::Union{AbstractArray, Nothing}; component_idx)
    
Evaluate marginals of the circuit bottom-up for given input(s).

Missing values should be denoted by `missing` in the data.

Outputs ``\\log{p(x^o)}`` for each data point. 
"""
marginal(root::ProbCircuit, data::Union{Real,Missing}...) =
    marginal(root, collect(Union{Bool,Missing}, data))

marginal(root::ProbCircuit, data::Union{Vector{Union{Bool,Missing}},CuVector{UInt8}}) =
    marginal(root, DataFrame(reshape(data, 1, :), :auto))[1]

marginal(circuit::ProbCircuit, data::Union{DataFrame, Vector{DataFrame}}) =
    marginal(same_device(ParamBitCircuit(circuit, data), data) , data)

function marginal(circuit::ParamBitCircuit, data::DataFrame)::AbstractVector
    marginal_all(circuit, data)[:,end]
end

"""
    marginal(circuit::SharedProbCircuit, data::DataFrame, weights::Union{AbstractArray, Nothing} = nothing; component_idx = 0)::AbstractVector

Computes marginals for one circuit in a `SharedProbCircuit`.
- `component_idx`: index of the SharedProbCircuit component to comptue marginals on.
"""
function marginal(circuit::SharedProbCircuit, data::DataFrame, weights::Union{AbstractArray, Nothing} = nothing; component_idx = 0)::AbstractVector
    if component_idx == 0
        @assert !isnothing(weights) "Missing weights for weighting mixture probabilities."
        @assert length(weights) == num_components(circuit) "Weights must have same dimension as number of components."
    else
        return MAR(ParamBitCircuit(circuit, data; component_idx = component_idx), data)
    end
    if weights isa AbstractVector weights = reshape(weights, (1, length(weights))) end
    n = length(weights)
    lls = Array{Float64, 2}(undef, num_examples(data), n)
    for i in 1:n
        pbc = ParamBitCircuit(circuit, data; component_idx = i)
        lls[:,i] .= MAR(pbc, data)
    end
    return logsumexp(lls .+ log.(weights), 2)
end

function marginal(circuit::ParamBitCircuit, data::Vector{DataFrame})::AbstractVector
    if isgpu(data)
        marginals = CuVector{Float64}(undef, num_examples(data))
    else
        marginals = Vector{Float64}(undef, num_examples(data))
    end
    
    v, start_idx = nothing, 1
    map(data) do d
        v = marginal_all(circuit, d, v)
        v_len = size(v, 1)
        @inbounds @views marginals[start_idx: start_idx + v_len - 1] .= v[:, end]
        start_idx += v_len
    end
    
    isgpu(data) ? convert(Array{Float64, 1}, to_cpu(marginals)) : marginals
end

"""
    MAR(pc, data) = marginal

Computes Marginal log likelhood of data. `MAR` is juat an alias for `marginal`. See docs for `marginal` for more details.
"""
const MAR = marginal

"""
    marginal_log_likelihood(pc, data)
    marginal_log_likelihood(pc, data, weights::DataFrame)
    marginal_log_likelihood(pc, data, weights::AbstractArray)
    marginal_log_likelihood(pc, data::Vector{DataFrame})
    
Compute the marginal likelihood of the PC given the data
"""
marginal_log_likelihood(pc, data) = begin
    if isweighted(data)
        # `data' is weighted according to its `weight' column
        data, weights = split_sample_weights(data)
        
        marginal_log_likelihood(pc, data, weights)
    else
        sum(marginal(pc, data))
    end
end

marginal_log_likelihood(pc, data, weights::DataFrame) = 
    marginal_log_likelihood(pc, data, weights[:, 1])

marginal_log_likelihood(pc, data, weights::AbstractArray) = begin
    if isgpu(weights)
        weights = to_cpu(weights)
    end
    likelihoods = marginal(pc, data)
    if isgpu(likelihoods)
        likelihoods = to_cpu(likelihoods)
    end
    mapreduce(*, +, likelihoods, weights)
end

marginal_log_likelihood(pbc::ParamBitCircuit, data::Vector{DataFrame}) = begin  
    total_ll::Float64 = 0.0
    if isweighted(data)
        data, weights = split_sample_weights(data)
        
        v = nothing
        for idx = 1 : length(data)
            v = marginal_all(pbc, data[idx], v)
            likelihoods = v[:,end]
            if isgpu(likelihoods)
                likelihoods = to_cpu(likelihoods)
            end
            w = weights[idx]
            if isgpu(w)
                w = to_cpu(w)
            end
            total_ll += mapreduce(*, +, likelihoods, w)
        end
    else
        v = nothing
        for idx = 1 : length(data)
            v = marginal_all(pbc, data[idx], v)
            likelihoods = v[:,end]
            total_ll += sum(isgpu(likelihoods) ? to_cpu(likelihoods) : likelihoods)
        end
    end
    
    if isgpu(data)
        CUDA.unsafe_free!(v) # save the GC some effort
    end
    
    total_ll
end

marginal_log_likelihood(pc, data::Vector{DataFrame}) = begin    
    pbc = ParamBitCircuit(pc, data)
    marginal_log_likelihood(pbc, data)
end

"""
    marginal_log_likelihood_avg(pc, data)

Compute the marginal likelihood of the PC given the data, averaged over all instances in the data. If the data is weighted then does a weighted average.
"""
marginal_log_likelihood_avg(pc, data) = begin
    if isweighted(data)
        # `data' is weighted according to its `weight' column
        data, weights = split_sample_weights(data)
        marginal_log_likelihood_avg(pc, data, weights)
    else
        marginal_log_likelihood(pc, data) / num_examples(data)
    end
end

marginal_log_likelihood_avg(pc, data, weights::DataFrame) =
    marginal_log_likelihood_avg(pc, data, weights[:, 1])
    
marginal_log_likelihood_avg(pc, data, weights) = begin
    if isgpu(weights)
        weights = to_cpu(weights)
    end
    marginal_log_likelihood(pc, data, weights)/sum(weights)
end

marginal_log_likelihood_avg(pc, data::Vector{DataFrame}) = begin
    total_ll = marginal_log_likelihood(pc, data)
    
    if isweighted(data)
        data, weights = split_sample_weights(data)
        
        total_weights = 0.0
        for idx = 1 : length(weights)
            total_weights += sum(isgpu(weights[idx]) ? to_cpu(weights[idx]) : weights[idx])
        end
        
        avg_ll = total_ll / total_weights
    else
        avg_ll = total_ll / num_examples(data)
    end
    
    avg_ll
end

#####################
# Circuit evaluation of *all* nodes in circuit
#####################

"Evaluate the probabilistic circuit bottom-up for a given input and return the marginal probability value of all nodes"
marginal_all(circuit::ProbCircuit, data::DataFrame) =
    marginal_all(same_device(ParamBitCircuit(circuit, data), data) , data)

function marginal_all(circuit::ParamBitCircuit, data, reuse=nothing)
    @assert num_features(data) == num_features(circuit) 
    values = init_marginal(data, reuse, num_nodes(circuit))
    marginal_layers(circuit, values)
    return values
end

"""
    init_marginal(data, reuse, num_nodes; Float=Float32)

Initialize values from the data (data frames)
"""
function init_marginal(data, reuse, num_nodes; Float=Float32)
    if isgpu(data)
        init_marginal_gpu(data, reuse, num_nodes; Float)
    else
        init_marginal_cpu(data, reuse, num_nodes; Float)
    end
end

"""
    init_marginal_cpu(data, reuse, num_nodes; Float)

Initialize values from the cpu data (data frames)    
"""
function init_marginal_cpu(data, reuse, num_nodes; Float)
    flowtype = Matrix{Float}
    values = similar!(reuse, flowtype, num_examples(data), num_nodes)
    @views values[:,LogicCircuits.TRUE_BITS]  .= log(one(Float))
    @views values[:,LogicCircuits.FALSE_BITS] .= log(zero(Float))
    nfeatures = num_features(data)
    for i=1:nfeatures
        values[:, 2+i] .= log.(coalesce.(data[:,i], one(Float)))
        values[:, 2+nfeatures+i] .= log.(coalesce.(one(Float) .- data[:,i], one(Float)))
    end
    return values
end

"""
    init_marginal_gpu(data, reuse, num_nodes; Float=Float32)
    
Initialize values from the gpu data
"""
function init_marginal_gpu(data, reuse, num_nodes; Float=Float32)
    flowtype = CuMatrix{Float}
    values = similar!(reuse, flowtype, num_examples(data), num_nodes)
    @views values[:, LogicCircuits.TRUE_BITS] .= log(one(Float))
    @views values[:, LogicCircuits.FALSE_BITS] .= log(zero(Float))

    nfeatures = num_features(data)
    
    # ## option 2 - still slow


    if data[!, 1] isa CuBitVector
        # Have to do this for CuBitVector since does not play well with CuArray broadcast kernels
        data_cpu = to_cpu(data)
        for i=1:nfeatures
            @views values[:, 2 + i]             .= to_gpu(log.(coalesce.(zero(Float) .+ data_cpu[!, i], one(Float))))
            @views values[:, 2 + nfeatures + i] .= to_gpu(log.(coalesce.(one(Float) .- data_cpu[!, i], one(Float))))
        end
    else
        for i=1:nfeatures
            @views values[:, 2 + i]             .= log.(coalesce.(zero(Float) .+ data[!, i], one(Float)))
            @views values[:, 2 + nfeatures + i] .= log.(coalesce.(one(Float) .- data[!, i], one(Float)))
        end
    end

    # Option 1 - very slow
    # data_cpu = to_cpu(data)
    # for i=1:nfeatures
    #     marg_pos::Vector{Float} = log.(coalesce.(data_cpu[:,i], one(Float)))
    #     marg_neg::Vector{Float} = log.(coalesce.(1.0 .- data_cpu[:,i], one(Float)))
    #     values[:,2+i] .= same_device(marg_pos, values)
    #     values[:,2+nfeatures+i] .= same_device(marg_neg, values)
    # end
    return values
end
# upward pass helpers on CPU

"Compute marginals on the CPU (SIMD & multi-threaded)"
function marginal_layers(circuit::ParamBitCircuit, values::Matrix)
    bc::BitCircuit = circuit.bitcircuit
    els = bc.elements
    pars = circuit.params
    for layer in bc.layers[2:end]
        Threads.@threads for dec_id in layer
            j = @inbounds bc.nodes[1,dec_id]
            els_end = @inbounds bc.nodes[2,dec_id]
            if j == els_end
                assign_marginal(values, dec_id, els[2,j], els[3,j], pars[j])
                j += 1
            else
                assign_marginal(values, dec_id, els[2,j], els[3,j], els[2,j+1], els[3,j+1], pars[j], pars[j+1])
                j += 2
            end
            while j <= els_end
                accum_marginal(values, dec_id, els[2,j], els[3,j], pars[j])
                j += 1
            end
        end
    end
end

assign_marginal(v::Matrix{<:AbstractFloat}, i, e1p, e1s, p1) =
    @views @. @avx v[:,i] = v[:,e1p] + v[:,e1s] + p1

accum_marginal(v::Matrix{<:AbstractFloat}, i, e1p, e1s, p1) = begin
    @avx for j=1:size(v,1)
        @inbounds x = v[j,i]
        @inbounds y = v[j,e1p] + v[j,e1s] + p1
        Δ = ifelse(x == y, zero(eltype(v)), abs(x - y))
        @inbounds  v[j,i] = max(x, y) + log1p(exp(-Δ))
    end
end

assign_marginal(v::Matrix{<:AbstractFloat}, i, e1p, e1s, e2p, e2s, p1, p2) = begin
    @avx for j=1:size(v,1)
        @inbounds x = v[j,e1p] + v[j,e1s] + p1
        @inbounds y = v[j,e2p] + v[j,e2s] + p2
        Δ = ifelse(x == y, zero(eltype(v)), abs(x - y))
        @inbounds  v[j,i] = max(x, y) + log1p(exp(-Δ))
    end
end

# upward pass helpers on GPU

"Compute marginals on the GPU"
function marginal_layers(circuit::ParamBitCircuit, values::CuMatrix)
    circuit = to_gpu(circuit)
    bc = circuit.bitcircuit
    CUDA.@sync for (li, layer) in enumerate(bc.layers[2:end])
        num_examples = size(values, 1)
        num_decision_sets = length(layer)

        kernel = @cuda name="marginal_layers_cuda" launch=false marginal_layers_cuda(layer, bc.nodes, bc.elements, circuit.params, values)
        config = launch_configuration(kernel.fun)
        threads, blocks =  balance_threads_2d(num_examples, num_decision_sets, config.threads)
        kernel(layer, bc.nodes, bc.elements, circuit.params, values; threads, blocks)
    end
end

"CUDA kernel for circuit evaluation"
function marginal_layers_cuda(layer, nodes, elements, params, values)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().y
    for j = index_x:stride_x:size(values,1)
        for i = index_y:stride_y:length(layer)
            decision_id = @inbounds layer[i]
            k = @inbounds nodes[1,decision_id]
            els_end = @inbounds nodes[2,decision_id]
            @inbounds x = values[j, elements[2,k]] + values[j, elements[3,k]] + params[k]
            while k < els_end
                k += 1
                @inbounds y = values[j, elements[2,k]] + values[j, elements[3,k]] + params[k]
                Δ = ifelse(x == y, zero(eltype(values)), CUDA.abs(x - y))
                x = max(x, y) + CUDA.log1p(CUDA.exp(-Δ))
            end
            values[j, decision_id] = x
        end
    end
    return nothing
end


#####################
# Bit circuit marginals and flows (up and downward pass)
#####################

"Compute the marginal and flow of each node"
function marginal_flows(circuit::ProbCircuit, data, 
    reuse_values=nothing, reuse_flows=nothing; on_node=noop, on_edge=noop, weights=nothing) 
    pbc = same_device(ParamBitCircuit(circuit, data), data)
    v, f = marginal_flows(pbc, data, reuse_values, reuse_flows; on_node, on_edge, weights)
    v, f, pbc.bitcircuit.node2id
end

function marginal_flows(circuit::ParamBitCircuit, data, 
            reuse_values=nothing, reuse_flows=nothing; on_node=noop, on_edge=noop, weights=nothing)
    @assert isgpu(data) == isgpu(circuit) "ParamBitCircuit and data need to be on the same device"
    values = marginal_all(circuit, data, reuse_values)
    flows = marginal_flows_down(circuit, values, reuse_flows; on_node, on_edge, weights)
    return values, flows
end

#####################
# Bit circuit marginal flows downward pass
#####################

"When marginals of nodes have already been computed, do a downward pass computing the marginal flows at each node"
function marginal_flows_down(circuit::ParamBitCircuit, values, reuse=nothing; on_node=noop, on_edge=noop, weights=nothing)
    flows = similar!(reuse, typeof(values), size(values)...)
    marginal_flows_down_layers(circuit, flows, values, on_node, on_edge; weights = weights)
    return flows
end

# downward pass helpers on CPU

"Evaluate marginals of the layers of a parameter bit circuit on the CPU (SIMD & multi-threaded)"
function marginal_flows_down_layers(pbc::ParamBitCircuit, flows::Matrix, values::Matrix, on_node, on_edge; weights = nothing)
    @assert flows !== values
    circuit::BitCircuit = pbc.bitcircuit 
    els = circuit.elements
    for layer in Iterators.reverse(circuit.layers)
        Threads.@threads for dec_id in layer
            par_start = @inbounds circuit.nodes[3,dec_id]
            if iszero(par_start)
                if dec_id == num_nodes(circuit)
                    # marginal flow start from 0.0
                    @inbounds @views flows[:, dec_id] .= zero(eltype(flows))
                end
                # no parents, ignore (can happen for false/true node and root)
            else
                par_end = @inbounds circuit.nodes[4,dec_id]
                for j = par_start:par_end
                    par = @inbounds circuit.parents[j]
                    grandpa = @inbounds els[1,par]
                    sib_id = sibling(els, par, dec_id)
                    single_child = has_single_child(circuit.nodes, grandpa)
                    if single_child
                        if j == par_start
                            @inbounds @views @. flows[:, dec_id] = flows[:, grandpa]
                        else
                            accum_marg_flow(flows, dec_id, grandpa)
                        end
                    else
                        θ = eltype(flows)(pbc.params[par])
                        if j == par_start
                            assign_marg_flow(flows, values, dec_id, grandpa, sib_id, θ)
                        else
                            accum_marg_flow(flows, values, dec_id, grandpa, sib_id, θ)
                        end
                    end
                    # report edge flow only once:
                    sib_id > dec_id && on_edge(flows, values, dec_id, sib_id, par, grandpa, single_child, weights)
                end
            end
            on_node(flows, values, dec_id, weights)
        end
    end
end

function assign_marg_flow(f::Matrix{<:AbstractFloat}, v, d, g, s, θ)
    @inbounds @simd for j=1:size(f,1) #@avx gives incorrect results
        edge_flow = v[j, s] + v[j, d] - v[j, g] + f[j, g] + θ
        edge_flow = ifelse(isnan(edge_flow), typemin(eltype(f)), edge_flow)
        f[j, d] = edge_flow 
    end
    # @assert !any(isnan, f[:,d])
end

function accum_marg_flow(f::Matrix{<:AbstractFloat}, d, g)
    @inbounds @simd for j=1:size(f,1) #@avx gives incorrect results
        x = f[j, d]
        y = f[j, g]
        Δ = ifelse(x == y, zero(eltype(f)), abs(x - y))
        f[j, d] = max(x, y) + log1p(exp(-Δ))
    end
    # @assert !any(isnan, f[:,d])
end

function accum_marg_flow(f::Matrix{<:AbstractFloat}, v, d, g, s, θ)
    @inbounds @simd for j=1:size(f,1) #@avx gives incorrect results
        x = f[j, d]
        y = v[j, s] + v[j, d] - v[j, g] + f[j, g] + θ
        y = ifelse(isnan(y), typemin(eltype(f)), y)
        Δ = ifelse(x == y, zero(eltype(f)), abs(x - y))
        f[j, d] = max(x, y) + log1p(exp(-Δ))
    end
    # @assert !any(isnan, f[:,d])
end

# downward pass helpers on GPU

"Pass marginal flows down the layers of a bit circuit on the GPU"
function marginal_flows_down_layers(pbc::ParamBitCircuit, flows::CuMatrix, values::CuMatrix, 
            on_node, on_edge; weights = nothing)
    bc = pbc.bitcircuit
    CUDA.@sync for (li, layer) in enumerate(Iterators.reverse(bc.layers))
        num_examples = size(values, 1)
        num_decision_sets = length(layer)
        
        kernel = @cuda name="marginal_flows_down_layers_cuda" launch=false marginal_flows_down_layers_cuda(layer, bc.nodes, bc.elements, bc.parents, pbc.params, flows, values, on_node, on_edge, weights)
        config = launch_configuration(kernel.fun)
        threads, blocks = balance_threads_2d(num_examples, num_decision_sets, config.threads)
        kernel(layer, bc.nodes, bc.elements, bc.parents, pbc.params, flows, values, on_node, on_edge, weights;
                threads=threads, blocks=blocks)
    end
end

"CUDA kernel for passing marginal flows down circuit"
function marginal_flows_down_layers_cuda(layer, nodes, elements, parents, params, flows, values, on_node, on_edge, weights::Nothing)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().y
    for k = index_x:stride_x:size(values,1)
        for i = index_y:stride_y:length(layer)
            dec_id = @inbounds layer[i]
            if dec_id == size(nodes,2)
                # populate root flows
                flow = zero(eltype(flows))
            else
                par_start = @inbounds nodes[3,dec_id]
                flow = typemin(eltype(flows)) # log(0)
                if !iszero(par_start)
                    par_end = @inbounds nodes[4,dec_id]
                    for j = par_start:par_end
                        par = @inbounds parents[j]
                        grandpa = @inbounds elements[1,par]
                        v_gp = @inbounds values[k, grandpa]
                        prime = elements[2,par]
                        sub = elements[3,par]
                        θ = eltype(flows)(params[par])
                        if !iszero(v_gp) # edge flow only gets reported when non-zero
                            f_gp = @inbounds flows[k, grandpa]
                            single_child = has_single_child(nodes, grandpa)
                            if single_child
                                edge_flow = f_gp
                            else
                                v_prime = @inbounds values[k, prime]
                                v_sub = @inbounds values[k, sub]
                                edge_flow = compute_marg_edge_flow(v_prime, v_sub, v_gp, f_gp, θ)
                            end
                            flow = logsumexp_cuda(flow, edge_flow)
                            # report edge flow only once:
                            dec_id == prime && on_edge(flows, values, prime, sub, par, grandpa, k, edge_flow, single_child, nothing)
                        end
                    end
                end
            end
            @inbounds flows[k, dec_id] = flow
            on_node(flows, values, dec_id, k, flow, nothing)
        end
    end
    return nothing
end
function marginal_flows_down_layers_cuda(layer, nodes, elements, parents, params, flows, values, on_node, on_edge, weights::Any)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().y
    for k = index_x:stride_x:size(values,1)
        weight = @inbounds weights[k]
        for i = index_y:stride_y:length(layer)
            dec_id = @inbounds layer[i]
            if dec_id == size(nodes,2)
                # populate root flows
                flow = zero(eltype(flows))
            else
                par_start = @inbounds nodes[3,dec_id]
                flow = typemin(eltype(flows)) # log(0)
                if !iszero(par_start)
                    par_end = @inbounds nodes[4,dec_id]
                    for j = par_start:par_end
                        par = @inbounds parents[j]
                        grandpa = @inbounds elements[1,par]
                        v_gp = @inbounds values[k, grandpa]
                        prime = elements[2,par]
                        sub = elements[3,par]
                        θ = eltype(flows)(params[par])
                        if !iszero(v_gp) # edge flow only gets reported when non-zero
                            f_gp = @inbounds flows[k, grandpa]
                            single_child = has_single_child(nodes, grandpa)
                            if single_child
                                edge_flow = f_gp
                            else
                                v_prime = @inbounds values[k, prime]
                                v_sub = @inbounds values[k, sub]
                                edge_flow = compute_marg_edge_flow(v_prime, v_sub, v_gp, f_gp, θ, weight)
                            end
                            flow = logsumexp_cuda(flow, edge_flow)
                            # report edge flow only once:
                            dec_id == prime && on_edge(flows, values, prime, sub, par, grandpa, k, edge_flow, single_child, weight)
                        end
                    end
                end
            end
            @inbounds flows[k, dec_id] = flow
            on_node(flows, values, dec_id, k, flow, weight)
        end
    end
    return nothing
end

@inline function compute_marg_edge_flow(p_up, s_up, n_up, n_down, θ)
    x = p_up + s_up - n_up + n_down + θ
    ifelse(isnan(x), typemin(n_down), x)
end
@inline function compute_marg_edge_flow(p_up, s_up, n_up, n_down, θ, weight)
    x = p_up + s_up - n_up + n_down + θ
    ifelse(isnan(x), typemin(n_down), x)
end