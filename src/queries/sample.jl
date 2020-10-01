export sample, to_sampled_dataframes

import DataFrames: DataFrame, mapcols!
import Random: default_rng

#####################
# Circuit sampling
#####################

"Sample states from the circuit distribution."
function sample(pc::ProbCircuit; rng = default_rng())
    states, prs = sample(pc, 1, [missing for i=1:num_variables(pc)]...; rng)
    return states[1,:], prs[1]
end

sample(pc::ProbCircuit, num_samples; rng = default_rng(), gpu=false) =
    sample(pc, num_samples, [missing for i=1:num_variables(pc)]...; rng, gpu)

sample(pc::ProbCircuit, num_samples, inputs::Union{Bool,Missing}...; 
        rng = default_rng(), gpu=false) =
    sample(pc, num_samples, collect(Union{Bool,Missing}, inputs); rng, gpu)

function sample(pc::ProbCircuit, num_samples, inputs::AbstractVector{Union{Bool,Missing}}; 
                rng = default_rng(), gpu=false)
    data = DataFrame(reshape(inputs, 1, :))
    data = gpu ? to_gpu(data) : data
    states, prs = sample(pc, num_samples, data; rng)
    return states[:,1,:], prs[:,1]
end

sample(circuit::ProbCircuit, num_samples, data::DataFrame; rng = default_rng()) =
    sample(same_device(ParamBitCircuit(circuit, data), data), num_samples, data; rng)

function sample(pbc::ParamBitCircuit, num_samples, data; Float = Float32, rng = default_rng())
    @assert isgpu(data) == isgpu(pbc) "ParamBitCircuit and data need to be on the same device"
    values = marginal_all(pbc, data)
    return sample_down(pbc, num_samples, data, values, rng, Float)
end

"Convert an array of samples into a vector of dataframes"
function to_sampled_dataframes(states) 
    dfs = mapslices(DataFrame, states, dims = [2,3])
    map(dfs) do df
        mapcols!(c -> BitVector(c), df)
    end
    return dfs
end

# CPU code

function sample_down(pbc, num_samples, data, values::Array, rng, ::Type{Float}) where Float
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
        edge_log_pr, prime, sub = sample_child_cpu(params, nodes, elements, ex_id, dec_id, values, rng)
        @inbounds logprob[s_id, ex_id] += edge_log_pr
        sample_rec(nl, params, nodes, elements, ex_id, s_id, prime, values, state, logprob, rng)
        sample_rec(nl, params, nodes, elements, ex_id, s_id, sub,   values, state, logprob, rng)
    end
end

function sample_child_cpu(params, nodes, elements, ex_id, dec_id, values, rng)
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


# GPU code

function sample_down(pbc, num_samples, data, values::CuArray, rng, ::Type{Float}) where Float
    CUDA.seed!(rand(rng, UInt))
    state = CUDA.zeros(Bool, num_samples, num_examples(data), num_features(data))
    logprob = CUDA.zeros(Float, num_samples, num_examples(data))
    stack = CUDA.zeros(Int32, num_samples, num_examples(data), num_features(data)+3)
    @inbounds stack[:,:,1] .= 1 # start with 1 dec_id in the stack
    @inbounds stack[:,:,2] .= num_nodes(pbc) # start with the pc in the stack
    num_threads =  balance_threads(num_samples, num_examples(data), 8)
    num_blocks = (ceil(Int, num_samples/num_threads[1]), 
                  ceil(Int, num_examples(data)/num_threads[2]))
    CUDA.@sync  while true
        r = CUDA.rand(num_samples, num_examples(data))
        @cuda threads=num_threads blocks=num_blocks sample_cuda_kernel(num_leafs(pbc), params(pbc), nodes(pbc), elements(pbc), values, state, logprob, stack, r, Float)
        all_empty(stack) && break
    end
    CUDA.unsafe_free!(values) # save the GC some effort
    return state, logprob
end

function sample_cuda_kernel(nl, params, nodes, elements, values, state, logprob, stack, r, ::Type{Float}) where Float
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().y
    for s_id = index_x:stride_x:size(state,1)
        for ex_id = index_y:stride_y:size(state,2)
            dec_id = pop_cuda!(stack, s_id, ex_id)
            if dec_id > zero(eltype(stack))
                if isleafgate(nl, dec_id)
                    if isliteralgate(nl, dec_id)
                        l = literal(nl, dec_id)
                        var = lit2var(l)
                        @inbounds state[s_id, ex_id, var] = (l > 0) 
                    end
                else
                    edge_log_pr, prime, sub = sample_child_cuda(params, nodes, elements, s_id, ex_id, dec_id, values, r, Float)
                    @inbounds logprob[s_id, ex_id] += edge_log_pr
                    push_cuda!(stack, prime, s_id, ex_id)
                    push_cuda!(stack, sub, s_id, ex_id)
                end
            end
        end
    end
    return nothing
end

function sample_child_cuda(params, nodes, elements, s_id, ex_id, dec_id, values, r, ::Type{Float}) where Float
    @inbounds els_start = nodes[1,dec_id]
    @inbounds els_end = nodes[2,dec_id]
    @inbounds threshold = CUDA.log(r[s_id, ex_id]) + values[ex_id, dec_id]
    cumul_prob::Float = -Inf
    j_sampled = els_end - els_start + 1 # give all numerical error probability to the last node
    for j = els_start:els_end
        @inbounds prime = elements[2,j]
        @inbounds sub = elements[3,j]
        @inbounds pr::Float = values[ex_id, prime] + values[ex_id, sub] + params[j]
        cumul_prob = logsumexp_cuda(cumul_prob, pr)
        if cumul_prob > threshold
            j_sampled = j
            break
        end
    end
    @inbounds return params[j_sampled], elements[2,j_sampled], elements[3,j_sampled] 
end