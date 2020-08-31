export EVI, log_likelihood_per_instance, log_likelihood, log_likelihood_avg

"""
Complete evidence queries
"""
# function log_likelihood_per_instance(pc::ProbCircuit, data)
#     @assert isbinarydata(data) "Can only calculate EVI on Bool data"
    
#     compute_flows(pc, data)
#     log_likelihoods = zeros(Float64, num_examples(data))
#     indices = init_array(Bool, num_examples(data))::BitVector
    
#     ll(n::ProbCircuit) = ()
#     ll(n::Union{PlainSumNode, StructSumNode}) = begin
#         if num_children(n) != 1 # other nodes have no effect on likelihood
#             foreach(children(n), n.log_probs) do c, log_theta
#                 indices = get_downflow(n, c)
#                 view(log_likelihoods, indices::BitVector) .+=  log_theta # see MixedProductKernelBenchmark.jl
#             end
#          end
#     end

#     foreach(ll, pc)
#     log_likelihoods
# end

function bitcircuit_with_params(pc, data)
    params::Vector{Float64} = Vector{Float64}()
    on_decision(n, cs, layer_id, decision_id, first_element, last_element) = begin
        if isnothing(n) # this decision node is not part of the PC
            # @assert first_element == last_element
            push!(params, 0.0)
        else
            # @assert last_element-first_element+1 == length(n.log_probs) "$last_element-$first_element+1 != $(length(n.log_probs))"
            append!(params, n.log_probs)
        end
    end
    bc = BitCircuit(pc, data; gpu = isgpu(data), on_decision)
    params = isgpu(data) ? to_gpu(params) : params
    (bc, params)
end

function log_likelihood_per_instance(pc::ProbCircuit, data)
    @assert isbinarydata(data) "Probabilistic circuit likelihoods are for binary data only"
    bc, params = bitcircuit_with_params(pc, data)
    if isgpu(data)
        log_likelihood_per_instance_gpu(bc, data, params)
    else
        log_likelihood_per_instance_cpu(bc, data, params)
    end
end

function log_likelihood_per_instance_cpu(bc, data, params)
    n_ex::Int = num_examples(data)
    ll::Vector{Float64} = zeros(Float64, n_ex)
    ll_lock::Threads.ReentrantLock = Threads.ReentrantLock()
    
    @inline function on_edge(flows, values, dec_id, el_id, p, s, els_start, els_end, locks)
        if els_start != els_end
            lock(ll_lock) do # TODO: move lock to inner loop?
                for i = 1:size(flows,1)
                    @inbounds edge_flow = values[i, p] & values[i, s] & flows[i, dec_id]
                    first_true_bit = trailing_zeros(edge_flow)+1
                    last_true_bit = 64-leading_zeros(edge_flow)
                    @simd for j= first_true_bit:last_true_bit
                        ex_id = ((i-1) << 6) + j
                        if get_bit(edge_flow, j)
                            @inbounds ll[ex_id] += params[el_id]
                        end
                    end
                end
            end
        end
        nothing
    end

    compute_values_flows(bc, data; on_edge)
    return ll
end

function log_likelihood_per_instance_gpu(bc, pseudocount)
    # node_counts::CuVector{Int32} = CUDA.zeros(Int32, num_nodes(bc))
    # edge_counts::CuVector{Int32} = CUDA.zeros(Int32, num_elements(bc))
    # params::CuVector{Float64} = CuVector{Float64}(undef, num_elements(bc))
    # # need to manually cudaconvert closure variables
    # node_counts_device = CUDA.cudaconvert(node_counts)
    # edge_counts_device = CUDA.cudaconvert(edge_counts)
    # params_device = CUDA.cudaconvert(params)
    
    # @inline function on_node(flows, values, dec_id, els_start, els_end, ex_id)
    #     if els_start != els_end
    #         @inbounds c::Int32 = count_ones(flows[ex_id, dec_id]) # cast for @atomic to be happy
    #         CUDA.@atomic node_counts_device[dec_id] += c
    #     end
    #     if isone(ex_id) # only do this once
    #         for i=els_start:els_end
    #             params_device[i] = pseudocount/(els_end-els_start+1)
    #         end
    #     end
    #     nothing
    # end
    
    # @inline function on_edge(flows, values, dec_id, el_id, p, s, els_start, els_end, ex_id, edge_flow)
    #     if els_start != els_end
    #         c::Int32 = count_ones(edge_flow) # cast for @atomic to be happy
    #         CUDA.@atomic edge_counts_device[el_id] += c
    #     end
    #     nothing
    # end

    # function get_params()
    #     parent_counts = @views node_counts[bc.elements[1,:]]
    #     params .= log.(params .+ edge_counts) .- log.(parent_counts .+ pseudocount)
    #     to_cpu(params)
    # end

    # return (on_node, on_edge, get_params)
end

EVI = log_likelihood_per_instance

log_likelihood(pc, data) = sum(log_likelihood_per_instance(pc, data))
log_likelihood_avg(pc, data) = log_likelihood(pc, data)/num_examples(data)