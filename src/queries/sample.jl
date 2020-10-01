export sample

import DataFrames: DataFrame, mapcols!
import Random: default_rng

#####################
# Circuit MAP/MPE evaluation
#####################

"Sample states from the circuit distribution"
function sample(pc::ProbCircuit; rng = default_rng())
    dfs, prs = sample(pc, 1, [missing for i=1:num_variables(pc)]...; rng)
    return dfs[1], prs[1]
end

sample(pc::ProbCircuit, num_samples; rng = default_rng()) =
    sample(pc, num_samples, [missing for i=1:num_variables(pc)]...; rng)

sample(pc::ProbCircuit, num_samples, data::Union{Bool,Missing}...; rng = default_rng()) =
    sample(pc, num_samples, collect(Union{Bool,Missing}, data); rng)

function sample(pc::ProbCircuit, num_samples, data::Union{Vector{Union{Bool,Missing}},CuVector{UInt8}}; rng = default_rng())
    dfs, prs = sample(pc, num_samples, DataFrame(reshape(data, 1, :)); rng)
    return map(df -> example(df,1), dfs), prs[:,1]
end

sample(circuit::ProbCircuit, num_samples, data::DataFrame; rng = default_rng()) =
    sample(same_device(ParamBitCircuit(circuit, data), data), num_samples, data; rng)

function sample(pbc::ParamBitCircuit, num_samples, data; Float=Float32, rng = default_rng())
    @assert isgpu(data) == isgpu(pbc) "ParamBitCircuit and data need to be on the same device"
    values = marginal_all(pbc, data)
    states, logprobs = sample_down(pbc, num_samples, data, values, rng, Float)
    if isgpu(values)
        # CUDA.unsafe_free!(values) # save the GC some effort
        # # do the conversion to a CuBitVector on the CPU...
        # df = DataFrame(to_cpu(state))
        # mapcols!(c -> to_gpu(BitVector(c)), df)
    else
        dfs = mapslices(DataFrame, states, dims = [2,3])
        map(dfs) do df
            mapcols!(c -> BitVector(c), df)
        end
    end
    dfs, logprobs
end

"Sample a child node id of a given decision node"
function sample_child(params, nodes, elements, ex_id, dec_id, values, rng)
    @inbounds els_start = nodes[1,dec_id]
    @inbounds els_end = nodes[2,dec_id]
    threshold = log(rand(rng)) + values[ex_id, dec_id]
    cumul_prob = -Inf
    j_sampled = els_end - els_start + 1 # give all numerical error probability to the last node
    for j = els_start:els_end
        @inbounds prime = elements[2,j]
        @inbounds sub = elements[3,j]
        @inbounds pr = values[ex_id, prime] + values[ex_id, sub] + params[j]
        Δ = ifelse(cumul_prob == pr, zero(cumul_prob), abs(cumul_prob - pr))
        cumul_prob = max(cumul_prob, pr) + log1p(exp(-Δ))
        if cumul_prob > threshold
            j_sampled = j
            break
        end
    end
    @inbounds return params[j_sampled], elements[2,j_sampled], elements[3,j_sampled] 
end

# CPU code

function sample_down(pbc, num_samples, data, values::Array, rng, Float)
    state = zeros(Bool, num_samples, num_examples(data), num_features(data))
    logprob = zeros(Float, num_samples, num_examples(data))
    Threads.@threads for (s_id, ex_id) = collect(Iterators.product(1:size(state,1), 1:size(state,2)))
        sample_rec(num_leafs(pbc), params(pbc), nodes(pbc), elements(pbc), ex_id, s_id, num_nodes(pbc), values, state, logprob, rng)
    end
    return state, logprob
end

function sample_rec(nl, params, nodes, elements, ex_id, s_id, dec_id, values, state, logprob, rng)
    if isleafgate(nl, dec_id)
        if isliteralgate(nl, dec_id)
            l = literal(nl, dec_id)
            @inbounds state[s_id, ex_id, lit2var(l)] = (l > 0) 
        end
    else
        edge_log_pr, prime, sub = sample_child(params, nodes, elements, ex_id, dec_id, values, rng)
        @inbounds logprob[s_id, ex_id] += edge_log_pr
        sample_rec(nl, params, nodes, elements, ex_id, s_id, prime, values, state, logprob, rng)
        sample_rec(nl, params, nodes, elements, ex_id, s_id, sub,   values, state, logprob, rng)
    end
end

# GPU code

# function sample_down(pbc, data, values::CuArray; Float=Float32)
#     state = CUDA.zeros(Bool, num_examples(data), num_features(data))
#     logprob = CUDA.zeros(Float, num_examples(data))
#     stack = CUDA.zeros(Int32, num_examples(data), num_features(data)+3)
#     @inbounds stack[:,1] .= 1 # start with 1 dec_id in the stack
#     @inbounds stack[:,2] .= num_nodes(pbc) # start with the pc in the stack
#     num_threads = 256
#     num_blocks = ceil(Int, size(state,1)/num_threads)
#     CUDA.@sync begin
#         @cuda threads=num_threads blocks=num_blocks sample_cuda_kernel(num_leafs(pbc), params(pbc), nodes(pbc), elements(pbc), values, state, logprob, stack)
#     end
#     return state, logprob
# end

# function sample_cuda_kernel(nl, params, nodes, elements, values, state, logprob, stack)
#     index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride_x = blockDim().x * gridDim().x
#     for ex_id = index_x:stride_x:size(state,1)
#         dec_id = pop_cuda!(stack, ex_id)
#         while dec_id > zero(eltype(stack))
#             if isleafgate(nl, dec_id)
#                 if isliteralgate(nl, dec_id)
#                     l = literal(nl, dec_id)
#                     var = lit2var(l)
#                     @inbounds state[ex_id, var] = (l > 0) 
#                 end
#             else
#                 edge_log_pr, prime, sub = sample_child(params, nodes, elements, ex_id, dec_id, values)
#                 @inbounds logprob[ex_id] += edge_log_pr
#                 push_cuda!(stack, ex_id, prime)
#                 push_cuda!(stack, ex_id, sub)
#             end
#             dec_id = pop_cuda!(stack, ex_id)
#         end
#     end
#     return nothing
# end