export estimate_parameters, uniform_parameters, estimate_parameters_em, test

using StatsFuns: logsumexp
using CUDA
using LoopVectorization

"""
Maximum likilihood estimation of parameters given data
"""
function estimate_parameters(pc::ProbCircuit, data; pseudocount::Float64)
    @assert isbinarydata(data) "Probabilistic circuit parameter estimation for binary data only"
    bc = BitCircuit(pc, data; reset=false)
    params = if isgpu(data)
        estimate_parameters_gpu(to_gpu(bc), data, pseudocount)
    else
        estimate_parameters_cpu(bc, data, pseudocount)
    end
    estimate_parameters_cached!(pc, bc, params)
    params
end

function estimate_parameters_cached!(pc, bc, params)
    foreach_reset(pc) do pn
        if is⋁gate(pn)
            if num_children(pn) == 1
                pn.log_probs .= zero(Float64)
            else
                id = (pn.data::⋁NodeIds).node_id
                @inbounds els_start = bc.nodes[1,id]
                @inbounds els_end = bc.nodes[2,id]
                @inbounds @views pn.log_probs .= params[els_start:els_end]
                @assert isapprox(sum(exp.(pn.log_probs)), 1.0, atol=1e-6) "Parameters do not sum to one locally: $(sum(exp.(pn.log_probs))); $(pn.log_probs)"
                pn.log_probs .-= logsumexp(pn.log_probs) # normalize away any leftover error
            end
        end
    end
    nothing
end

function estimate_parameters_cpu(bc::BitCircuit, data, pseudocount)
    # no need to synchronize, since each computation is unique to a decision node
    node_counts::Vector{UInt} = Vector{UInt}(undef, num_nodes(bc))
    log_params::Vector{Float64} = Vector{Float64}(undef, num_elements(bc))

    @inline function on_node(flows, values, dec_id)
        node_counts[dec_id] = sum(1:size(flows,1)) do i
            count_ones(flows[i, dec_id]) 
        end
    end

    @inline function estimate(element, decision, edge_count)
        num_els = num_elements(bc.nodes, decision)
        log_params[element] = 
            log((edge_count+pseudocount/num_els)
                   /(node_counts[decision]+pseudocount))
    end

    @inline function on_edge(flows, values, prime, sub, element, grandpa, single_child)
        if !single_child
            edge_count = sum(1:size(flows,1)) do i
                count_ones(values[i, prime] & values[i, sub] & flows[i, grandpa]) 
            end
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end

    v, f = satisfies_flows(bc, data; on_node, on_edge)

    return log_params
end

function estimate_parameters_gpu(bc::BitCircuit, data, pseudocount)
    node_counts::CuVector{Int32} = CUDA.zeros(Int32, num_nodes(bc))
    edge_counts::CuVector{Int32} = CUDA.zeros(Int32, num_elements(bc))
    # need to manually cudaconvert closure variables
    node_counts_device = CUDA.cudaconvert(node_counts)
    edge_counts_device = CUDA.cudaconvert(edge_counts)
    
    @inline function on_node(flows, values, dec_id, chunk_id, flow)
        c::Int32 = CUDA.count_ones(flow) # cast for @atomic to be happy
        CUDA.@atomic node_counts_device[dec_id] += c
    end

    @inline function on_edge(flows, values, prime, sub, element, grandpa, chunk_id, edge_flow, single_child)
        if !single_child
            c::Int32 = CUDA.count_ones(edge_flow) # cast for @atomic to be happy
            CUDA.@atomic edge_counts_device[element] += c
        end
    end

    v, f = satisfies_flows(bc, data; on_node, on_edge)

    CUDA.unsafe_free!(v) # save the GC some effort
    CUDA.unsafe_free!(f) # save the GC some effort

    # TODO: reinstate simpler implementation once https://github.com/JuliaGPU/GPUArrays.jl/issues/313 is fixed and released
    @inbounds parents = bc.elements[1,:]
    @inbounds parent_counts = node_counts[parents]
    @inbounds parent_elcount = bc.nodes[2,parents] .- bc.nodes[1,parents] .+ 1 
    params = log.((edge_counts .+ (pseudocount ./ parent_elcount)) 
                    ./ (parent_counts .+ pseudocount))
    return to_cpu(params)
end

"""
Uniform distribution
"""
function uniform_parameters(pc::ProbCircuit)
    foreach(pc) do pn
        if is⋁gate(pn)
            if num_children(pn) == 1
                pn.log_probs .= 0.0
            else
                pn.log_probs .= log.(ones(Float64, num_children(pn)) ./ num_children(pn))
            end
        end
    end
end

"""
Expectation maximization parameter learning given missing data
"""
function estimate_parameters_em(pc::ProbCircuit, data; pseudocount::Float64)
    pbc = ParamBitCircuit(pc, data; reset=false)
    params = if isgpu(data)
        estimate_parameters_gpu(to_gpu(pbc), data, pseudocount)
    else
        estimate_parameters_cpu(pbc, data, pseudocount)
    end
    estimate_parameters_cached!(pc, pbc.bitcircuit, params)
    params
end

function estimate_parameters_cpu(pbc::ParamBitCircuit, data, pseudocount)
    # no need to synchronize, since each computation is unique to a decision node
    bc = pbc.bitcircuit
    node_counts::Vector{Float64} = Vector{Float64}(undef, num_nodes(bc))
    log_params::Vector{Float64} = Vector{Float64}(undef, num_elements(bc))

    @inline function on_node(flows, values, dec_id)
        sum_flows = map(1:size(flows,1)) do i
            flows[i, dec_id]
        end
        node_counts[dec_id] = logsumexp(sum_flows)
    end

    @inline function estimate(element, decision, edge_count)
        num_els = num_elements(bc.nodes, decision)
        log_params[element] = 
            log((exp(edge_count)+pseudocount/num_els) / (exp(node_counts[decision])+pseudocount))
    end

    @inline function on_edge(flows, values, prime, sub, element, grandpa, single_child)
        θ = eltype(flows)(pbc.params[element])
        if !single_child
            edge_flows = map(1:size(flows,1)) do i
                values[i, prime] + values[i, sub] - values[i, grandpa] + flows[i, grandpa] + θ
            end
            edge_count = logsumexp(edge_flows)
            estimate(element, grandpa, edge_count)
        end # no need to estimate single child params, they are always prob 1
    end

    v, f = marginal_flows(pbc, data; on_node, on_edge)

    return log_params
end
