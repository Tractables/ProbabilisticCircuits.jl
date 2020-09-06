using StatsFuns: logsumexp, log1pexp

using CUDA: CUDA, @cuda
using DataFrames: DataFrame
using LoopVectorization: @avx
using LogicCircuits: balance_threads

export marginal, marginal_all

#####################
# Circuit marginal evaluation
#####################
  
# evaluate a probabilistic circuit as a function
function (root::ProbCircuit)(data...)
    marginal(root, data...)
end

"Evaluate marginals of the circuit bottom-up for a given input"
marginal(root::ProbCircuit, data::Union{Real,Missing}...) =
    marginal(root, collect(Union{Bool,Missing}, data))

marginal(root::ProbCircuit, data::Union{Vector{Union{Bool,Missing}},CuVector{UInt8}}) =
    marginal(root, DataFrame(reshape(data, 1, :)))[1]

marginal(circuit::ProbCircuit, data::DataFrame) =
    marginal(same_device(ParamBitCircuit(circuit, data), data) , data)

function marginal(circuit::ParamBitCircuit, data::DataFrame)::AbstractVector
    marginal_all(circuit,data)[:,end]
end

#####################
# Circuit evaluation of *all* nodes in circuit
#####################

"Evaluate the probabilistic circuit bottom-up for a given input and return the marginal probability value of all nodes"
function marginal_all(circuit::ParamBitCircuit, data, reuse=nothing)
    @assert num_features(data) == num_features(circuit) 
    @assert isbinarydata(data)
    values = init_marginal(data, reuse, num_nodes(circuit))
    marginal_layers(circuit, values)
    return values
end

"Initialize values from the data (data frames)"
function init_marginal(data, reuse, num_nodes)
    flowtype = isgpu(data) ? CuMatrix{Float64} : Matrix{Float64}
    values = similar!(reuse, flowtype, num_examples(data), num_nodes)
    @views values[:,LogicCircuits.TRUE_BITS] .= log(one(Float64))
    @views values[:,LogicCircuits.FALSE_BITS] .= log(zero(Float64))
    # here we should use a custom CUDA kernel to extract Float marginals from bit vectors
    # for now the lazy solution is to move everything to the CPU and do the work there...
    data_cpu = to_cpu(data)
    for i=1:num_features(data)
        marg_pos::Vector{Float64} = log.(coalesce.(data_cpu[:,i], 1.0))
        marg_neg::Vector{Float64} = log.(coalesce.(1.0 .- data_cpu[:,i], 1.0))
        values[:,2+i] .= same_device(marg_pos, values)
        values[:,2+num_features(data)+i] .= same_device(marg_neg, values)
    end
    return values
end

# upward pass helpers on CPU

"Compute marginals on the CPU (SIMD & multi-threaded)"
function marginal_layers(circuit::ParamBitCircuit, values::Matrix)
    bc = circuit.bitcircuit
    els = bc.elements
    pars = circuit.params
    for layer in bc.layers
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
        Δ = ifelse(x == y, zero(Float64), abs(x - y))
        @inbounds  v[j,i] = max(x, y) + log1p(exp(-Δ))
    end
end

assign_marginal(v::Matrix{<:AbstractFloat}, i, e1p, e1s, e2p, e2s, p1, p2) = begin
    @avx for j=1:size(v,1)
        @inbounds x = v[j,e1p] + v[j,e1s] + p1
        @inbounds y = v[j,e2p] + v[j,e2s] + p2
        Δ = ifelse(x == y, zero(Float64), abs(x - y))
        @inbounds  v[j,i] = max(x, y) + log1p(exp(-Δ))
    end
end

# upward pass helpers on GPU

"Compute marginals on the GPU"
function marginal_layers(circuit::ParamBitCircuit, values::CuMatrix;  dec_per_thread = 8, log2_threads_per_block = 8)
    bc = circuit.bitcircuit
    CUDA.@sync for layer in bc.layers
        num_examples = size(values, 1)
        num_decision_sets = length(layer)/dec_per_thread
        num_threads =  balance_threads(num_examples, num_decision_sets, log2_threads_per_block)
        num_blocks = (ceil(Int, num_examples/num_threads[1]), 
                      ceil(Int, num_decision_sets/num_threads[2]))
        @cuda threads=num_threads blocks=num_blocks marginal_layers_cuda(layer, bc.nodes, bc.elements, circuit.params, values)
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
                Δ = ifelse(x == y, zero(Float64), CUDA.abs(x - y))
                x = max(x, y) + CUDA.log1p(CUDA.exp(-Δ))
            end
            values[j, decision_id] = x
        end
    end
    return nothing
end



#####################
# Bit circuit values and flows (up and downward pass)
#####################

"Compute the value and flow of each node"
function marginal_flows(circuit::ProbCircuit, data, 
    reuse_values=nothing, reuse_flows=nothing; on_node=noop, on_edge=noop) 
    bc = same_device(ProbBitCircuit(circuit, data), data)
    marginal_flows(bc, data, reuse_values, reuse_flows; on_node, on_edge)
end

function marginal_flows(circuit::ParamBitCircuit, data, 
            reuse_values=nothing, reuse_flows=nothing; on_node=noop, on_edge=noop)
    values = marginal_all(circuit, data, reuse_values)
    flows = marginal_flows_down(circuit, values, reuse_flows; on_node, on_edge)
    return values, flows
end

#####################
# Bit circuit flows downward pass
#####################

"When marginals at nodes have already been computed, do a downward pass computing the marginal flows at each node"
function marginal_flows_down(circuit::ParamBitCircuit, values, reuse=nothing; on_node=noop, on_edge=noop)
    flows = similar!(reuse, typeof(values), size(values)...)
    init_marginal_flows(flows, values)
    marginal_flows_down_layers(circuit, flows, values, on_node, on_edge)
    return flows
end

function init_marginal_flows(flows::AbstractArray{F}, values::AbstractArray{F}) where F
    flows .= zero(F)
    @views flows[:,end] .= values[:,end] # set flow at root
end

# downward pass helpers on CPU

function marginal_flows_down_layers(circuit::ParamBitCircuit, flows::Matrix, values::Matrix, on_node, on_edge)
    els = circuit.bitcircuit.elements
    locks = [Threads.ReentrantLock() for i=1:num_nodes(circuit)]    
    for layer in Iterators.reverse(circuit.bitcircuit.layers)
        Threads.@threads for dec_id in layer
            els_start = @inbounds circuit.bitcircuit.nodes[1,dec_id]
            els_end = @inbounds circuit.bitcircuit.nodes[2,dec_id]
            on_node(flows, values, dec_id, els_start, els_end, locks)
            #TODO do something faster when els_start == els_end?
            for j = els_start:els_end
                p = els[2,j]
                s = els[3,j]
                θ = circuit.params[j]
                accum_marginal_flow(flows, values, dec_id, p, s, θ, locks)
                on_edge(flows, values, dec_id, j, p, s, els_start, els_end, locks)
            end
        end
    end
end

function accum_marginal_flow(f::Matrix{<:AbstractFloat}, v, d, p, s, θ, locks)
    # retrieve locks in index order to avoid deadlock
    l1, l2 = order_asc(p,s)
    lock(locks[l1]) do 
        lock(locks[l2]) do 
            # note: in future, if there is a need to scale to many more threads, it would be beneficial to avoid this synchronization by ordering downward pass layers by child id, not parent id, so that there is no contention when processing a single layer and no need for synchronization, as in the upward pass
            @avx for j in 1:size(f,1)
                edge_flow = v[j, p] + v[j, s] - v[j, d] + f[j, d] + θ
                edge_flow = vifelse(isfinite(edge_flow), edge_flow, log(zero(Float32)))
                x = f[j, p]
                y = edge_flow
                Δ = ifelse(x == y, zero(Float64), abs(x - y))
                @inbounds f[j,p] = max(x, y) + log1p(exp(-Δ))
                x = f[j, s]
                Δ = ifelse(x == y, zero(Float64), abs(x - y))
                @inbounds f[j,s] = max(x, y) + log1p(exp(-Δ))
            end
        end
    end
end


# downward pass helpers on GPU

"Pass flows down the layers of a bit circuit on the GPU"
function marginal_flows_down_layers(circuit::ParamBitCircuit, flows::CuMatrix, values::CuMatrix, on_node, on_edge; 
            dec_per_thread = 4, log2_threads_per_block = 8)
    bc = circuit.bitcircuit
    CUDA.@sync for layer in Iterators.reverse(bc.layers)
        num_examples = size(values, 1)
        num_decision_sets = length(layer)/dec_per_thread
        num_threads =  balance_threads(num_examples, num_decision_sets, log2_threads_per_block)
        num_blocks = (ceil(Int, num_examples/num_threads[1]), 
                      ceil(Int, num_decision_sets/num_threads[2])) 
        @cuda threads=num_threads blocks=num_blocks marginal_flows_down_layers_cuda(layer, bc.nodes, bc.elements, circuit.params, flows, values, on_node, on_edge)
    end
end

"CUDA kernel for passing flows down circuit"
function marginal_flows_down_layers_cuda(layer, nodes, elements, params, flows, values, on_node, on_edge)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().y
    for k = index_x:stride_x:size(values,1)
        for i = index_y:stride_y:length(layer) #TODO swap loops??
            dec_id = @inbounds layer[i]
            els_start = @inbounds nodes[1,dec_id]
            els_end = @inbounds nodes[2,dec_id]
            n_up = @inbounds values[k, dec_id]
            on_node(flows, values, dec_id, els_start, els_end, k)
            if !iszero(n_up) # on_edge will only get called when edge flows are non-zero
                n_down = @inbounds flows[k, dec_id]
                #TODO do something faster when els_start == els_end?
                for j = els_start:els_end
                    p = @inbounds elements[2,j]
                    s = @inbounds elements[3,j]
                    @inbounds edge_flow = values[k, p] + values[k, s] - n_up + n_down + params[j]
                    # following needs to be memory safe, hence @atomic

                    @inbounds y = values[j, elements[2,k]] + values[j, elements[3,k]] + params[k]
                    Δ = ifelse(x == y, zero(Float64), CUDA.abs(x - y))
                    x = max(x, y) + CUDA.log1p(CUDA.exp(-Δ))

                    
                    accum_flow(flows, k, p, edge_flow)
                    accum_flow(flows, k, s, edge_flow)


                    
                    on_edge(flows, values, dec_id, j, p, s, els_start, els_end, k, edge_flow)
                end
            end
        end
    end
    return nothing
end

compute_edge_flow(p_up::AbstractFloat, s_up, n_up, n_down) = p_up * s_up / n_up * n_down
compute_edge_flow(p_up::Unsigned, s_up, n_up, n_down) = p_up & s_up & n_down

accum_flow(flows, j, e, edge_flow::AbstractFloat) = 
    CUDA.@atomic flows[j, e] += edge_flow #atomic is automatically inbounds

accum_flow(flows, j, e, edge_flow::Unsigned) = 
    CUDA.@atomic flows[j, e] |= edge_flow #atomic is automatically inbounds
