export estimate_parameters, uniform_parameters

using StatsFuns: logsumexp
using CUDA
using LoopVectorization

"""
Maximum likilihood estimation of parameters given data
"""
function estimate_parameters(pc::ProbCircuit, data; pseudocount::Float64)
    @assert isbinarydata(data)
    compute_flows(pc, data)
    foreach(pc) do pn
        if is⋁gate(pn)
            if num_children(pn) == 1
                pn.log_thetas .= 0.0
            else
                smoothed_flow = Float64(sum(get_downflow(pn))) + pseudocount
                uniform_pseudocount = pseudocount / num_children(pn)
                children_flows = map(c -> sum(get_downflow(pn, c)), children(pn))
                @. pn.log_thetas = log((children_flows + uniform_pseudocount) / smoothed_flow)
                @assert isapprox(sum(exp.(pn.log_thetas)), 1.0, atol=1e-6) "Parameters do not sum to one locally"
                # normalize away any leftover error
                pn.log_thetas .-= logsumexp(pn.log_thetas)
            end
        end
    end
end

function estimate_parameters2(pc::ProbCircuit, data; pseudocount::Float64)
    @assert isbinarydata(data) "Probabilistic circuit parameter estimation for binary data only"
    bc = BitCircuit(pc, data; reset=false) #TODO: avoid resetting bit, do it at the end when collecting results
    log_params, node_flow, edge_flow = if isgpu(data)
        error("todo")
    else
        estimate_parameters2_cpu(bc, pseudocount)
    end
    compute_values_flows(bc, data; node_flow, edge_flow)
    log_params_cpu::Vector{Float64} = to_cpu(log_params)
    foreach_reset(pc) do pn
        if is⋁gate(pn)
            if num_children(pn) == 1
                pn.log_thetas .= 0.0
            else
                id = (pn.data::NodeId).node_id
                @inbounds els_start = bc.nodes[1,id]
                @inbounds els_end = bc.nodes[2,id]
                @inbounds @views pn.log_thetas .= log_params_cpu[els_start:els_end]
                @assert isapprox(sum(exp.(pn.log_thetas)), 1.0, atol=1e-6) "Parameters do not sum to one locally"
                pn.log_thetas .-= logsumexp(pn.log_thetas) # normalize away any leftover error
            end
        end
    end
end

function estimate_parameters2_cpu(bc, pseudocount)
    # no need to synchronize, since each computation is unique to a decision node
    node_counts::Vector{UInt} = Vector{UInt}(undef, num_nodes(bc))
    log_params::Vector{Float64} = Vector{Float64}(undef, num_elements(bc))

    @inline function node_flow_cpu(flows, values, dec_id, els_start, els_end, locks)
        if els_start != els_end
            @inbounds node_counts[dec_id] = sum(1:size(flows,1)) do i
                count_ones(flows[i, dec_id]) 
            end
        end
        nothing
    end
    
    @inline function edge_flow_cpu(flows, values, dec_id, el_id, p, s, els_start, els_end, locks)
        log_theta = if els_start != els_end
            edge_count = sum(1:size(flows,1)) do i
                @inbounds count_ones(values[i, p] & values[i, s] & flows[i, dec_id]) 
            end
            log((edge_count+pseudocount/(els_end-els_start+1))/(node_counts[dec_id]+pseudocount))
        else
            zero(Float64)
        end
        @inbounds log_params[el_id] = log_theta
        nothing
    end

    return (log_params, node_flow_cpu, edge_flow_cpu)
end

"""
Uniform distribution
"""
function uniform_parameters(pc::ProbCircuit)
    foreach(pc) do pn
        if is⋁gate(pn)
            if num_children(pn) == 1
                pn.log_thetas .= 0.0
            else
                pn.log_thetas .= log.(ones(Float64, num_children(pn)) ./ num_children(pn))
            end
        end
    end
end

# TODO add em paramaters learning 