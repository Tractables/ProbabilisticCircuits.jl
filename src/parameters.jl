export estimate_parameters, uniform_parameters

using StatsFuns: logsumexp
using CUDA
using LoopVectorization

"""
Maximum likilihood estimation of parameters given data
"""
function estimate_parameters(pc::ProbCircuit, data; pseudocount::Float64)
    @assert isbinarydata(data) "Probabilistic circuit parameter estimation for binary data only"
    bc = BitCircuit(pc, data; reset=false)
    params::Vector{Float64} = if isgpu(data)
        estimate_parameters_gpu(to_gpu(bc), data, pseudocount)
    else
        estimate_parameters_cpu(bc, data, pseudocount)
    end
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
    params
end

function estimate_parameters_cpu(bc, data, pseudocount)
    # no need to synchronize, since each computation is unique to a decision node
    node_counts::Vector{UInt} = Vector{UInt}(undef, num_nodes(bc))
    log_params::Vector{Float64} = Vector{Float64}(undef, num_elements(bc))

    @inline function on_node(flows, values, dec_id)
        if !has_single_child(bc.nodes, dec_id)
            @inbounds node_counts[dec_id] = sum(1:size(flows,1)) do i
                count_ones(flows[i, dec_id]) 
            end
        end
    end

    @inline function estimate(el_id, parent, edge_count)
        num_els = num_elements(bc.nodes, parent)
        @inbounds log_params[el_id] = 
            log((edge_count+pseudocount/num_els)
                   /(node_counts[parent]+pseudocount))
    end
    
    @inline function on_edge(flows, values, dec_id, par, grandpa)
        if !has_single_child(bc.nodes, grandpa)
            edge_count = sum(1:size(flows,1)) do i
                @inbounds count_ones(flows[i, grandpa]) 
            end
            estimate(par, grandpa, edge_count)
        end
    end

    @inline function on_edge(flows, values, dec_id, par, grandpa, sib_id)
        if !has_single_child(bc.nodes, grandpa)
            edge_count = sum(1:size(flows,1)) do i
                @inbounds count_ones(values[i, dec_id] & values[i, sib_id] & flows[i, grandpa]) 
            end
            estimate(par, grandpa, edge_count)
        end
    end

    satisfies_flows(bc, data; on_node, on_edge)

    return log_params
end

function estimate_parameters_gpu(bc, data, pseudocount)
    node_counts::CuVector{Int32} = CUDA.zeros(Int32, num_nodes(bc))
    edge_counts::CuVector{Int32} = CUDA.zeros(Int32, num_elements(bc))
    params::CuVector{Float64} = CuVector{Float64}(undef, num_elements(bc))
    # need to manually cudaconvert closure variables
    node_counts_device = CUDA.cudaconvert(node_counts)
    edge_counts_device = CUDA.cudaconvert(edge_counts)
    params_device = CUDA.cudaconvert(params)
    
    @inline function on_node(flows, values, dec_id, els_start, els_end, ex_id)
        if els_start != els_end
            @inbounds c::Int32 = count_ones(flows[ex_id, dec_id]) # cast for @atomic to be happy
            CUDA.@atomic node_counts_device[dec_id] += c
        end
        if isone(ex_id) # only do this once
            for i=els_start:els_end
                @inbounds params_device[i] = pseudocount/(els_end-els_start+1)
            end
        end
        nothing
    end
    
    @inline function on_edge(flows, values, dec_id, el_id, p, s, els_start, els_end, ex_id, edge_flow)
        if els_start != els_end
            c::Int32 = count_ones(edge_flow) # cast for @atomic to be happy
            CUDA.@atomic edge_counts_device[el_id] += c
        end
        nothing
    end

    v, f = satisfies_flows(bc, data; on_node, on_edge)
    CUDA.unsafe_free!(v) # save the GC some effort
    CUDA.unsafe_free!(f) # save the GC some effort

    # TODO: reinstate following implementation once https://github.com/JuliaGPU/GPUArrays.jl/issues/313 is fixed and released
    # parent_counts = @views node_counts[bc.elements[1,:]]
    @inbounds parents = bc.elements[1,:]
    @inbounds parent_counts = node_counts[parents]
    params .= log.(params .+ edge_counts) .- log.(parent_counts .+ pseudocount)
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

# TODO add em paramaters learning 