export estimate_parameters, uniform_parameters

using StatsFuns: logsumexp
using CUDA
using LoopVectorization

"""
Maximum likilihood estimation of parameters given data
"""
function estimate_parameters(pc::ProbCircuit, data; pseudocount::Float64)
    @assert isbinarydata(data) "Probabilistic circuit parameter estimation for binary data only"
    bc = BitCircuit(pc, data; reset=false, gpu = isgpu(data))
    on_node, on_edge, get_params = if isgpu(data)
        estimate_parameters_gpu(bc, pseudocount)
    else
        estimate_parameters_cpu(bc, pseudocount)
    end
    compute_values_flows(bc, data; on_node, on_edge)
    params::Vector{Float64} = get_params()
    foreach_reset(pc) do pn
        if is⋁gate(pn)
            if num_children(pn) == 1
                pn.log_probs .= zero(Float64)
            else
                id = (pn.data::⋁NodeId).node_id
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

function estimate_parameters_cpu(bc, pseudocount)
    # no need to synchronize, since each computation is unique to a decision node
    node_counts::Vector{UInt} = Vector{UInt}(undef, num_nodes(bc))
    log_params::Vector{Float64} = Vector{Float64}(undef, num_elements(bc))

    @inline function on_node(flows, values, dec_id, els_start, els_end, locks)
        if els_start != els_end
            @inbounds node_counts[dec_id] = sum(1:size(flows,1)) do i
                count_ones(flows[i, dec_id]) 
            end
        end
        nothing
    end
    
    @inline function on_edge(flows, values, dec_id, el_id, p, s, els_start, els_end, locks)
        if els_start != els_end
            edge_count = sum(1:size(flows,1)) do i
                @inbounds count_ones(values[i, p] & values[i, s] & flows[i, dec_id]) 
            end
            # TODO do the log before the division?
            log_param = log((edge_count+pseudocount/(els_end-els_start+1))
                            /(node_counts[dec_id]+pseudocount))
            @inbounds log_params[el_id] = log_param
        end
        nothing
    end

    get_params() = log_params

    return (on_node, on_edge, get_params)
end

function estimate_parameters_gpu(bc, pseudocount)
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
                params_device[i] = pseudocount/(els_end-els_start+1)
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

    function get_params()
        parent_counts = @views node_counts[bc.elements[1,:]]
        params .= log.(params .+ edge_counts) .- log.(parent_counts .+ pseudocount)
        to_cpu(params)
    end

    return (on_node, on_edge, get_params)
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