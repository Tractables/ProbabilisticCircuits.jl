export max_a_posteriori, MAP

import DataFrames: DataFrame, mapcols!

#####################
# Circuit MAP/MPE evaluation
#####################

"Evaluate maximum a-posteriori state of the circuit for a given input"
max_a_posteriori(root::ProbCircuit, data::Union{Bool,Missing}...) =
    max_a_posteriori(root, collect(Union{Bool,Missing}, data))

max_a_posteriori(root::ProbCircuit, data::Union{Vector{Union{Bool,Missing}},CuVector{UInt8}}) =
    max_a_posteriori(root, DataFrame(reshape(data, 1, :)))[1,:]

max_a_posteriori(circuit::ProbCircuit, data::DataFrame) =
    max_a_posteriori(same_device(ParamBitCircuit(circuit, data), data), data)

function max_a_posteriori(pbc::ParamBitCircuit, data; Float=Float32)
    @assert isgpu(data) == isgpu(pbc) "ParamBitCircuit and data need to be on the same device"
    values = marginal_all(pbc, data)
    state, logprob = map_down(pbc, data, values; Float)
    if isgpu(values)
        CUDA.unsafe_free!(values) # save the GC some effort
        # do the conversion to a CuBitVector on the CPU...
        df = DataFrame(to_cpu(state))
        mapcols!(c -> to_gpu(BitVector(c)), df)
    else
        df = DataFrame(state)
        mapcols!(c -> BitVector(c), df)
    end
    df, logprob
end

"""
Maximum a-posteriori queries
"""
const MAP = max_a_posteriori

"""
Mode of the distribution
"""
const mode = max_a_posteriori

"Find the MAP child value and node id of a given decision node"
function map_child(params, nodes, elements, ex_id, dec_id, values)
    @inbounds els_start = nodes[1,dec_id]
    @inbounds els_end = nodes[2,dec_id]
    pr_opt = typemin(eltype(values))
    j_opt = 1
    for j = els_start:els_end
        @inbounds prime = elements[2,j]
        @inbounds sub = elements[3,j]
        @inbounds pr = values[ex_id, prime] + values[ex_id, sub] + params[j]
        if pr > pr_opt
            pr_opt = pr
            j_opt = j
        end
    end
    @inbounds return params[j_opt], elements[2,j_opt], elements[3,j_opt] 
end

# CPU code

function map_down(pbc, data, values::Array; Float=Float32)
    state = zeros(Bool, num_examples(data), num_features(data))
    logprob = zeros(Float, num_examples(data))
    Threads.@threads for ex_id = 1:size(state,1)
        map_rec(num_leafs(pbc), params(pbc), nodes(pbc), elements(pbc), ex_id, num_nodes(pbc), values, state, logprob)
    end
    return state, logprob
end

function map_rec(nl, params, nodes, elements, ex_id, dec_id, values, state, logprob)
    if isleafgate(nl, dec_id)
        if isliteralgate(nl, dec_id)
            l = literal(nl, dec_id)
            @inbounds state[ex_id, lit2var(l)] = (l > 0) 
        end
    else
        edge_log_pr, prime, sub = map_child(params, nodes, elements, ex_id, dec_id, values)
        @inbounds logprob[ex_id] += edge_log_pr
        map_rec(nl, params, nodes, elements, ex_id, prime, values, state, logprob)
        map_rec(nl, params, nodes, elements, ex_id, sub,   values, state, logprob)
    end
end

# GPU code

function map_down(pbc, data, values::CuArray; Float=Float32)
    state = CUDA.zeros(Bool, num_examples(data), num_features(data))
    logprob = CUDA.zeros(Float, num_examples(data))
    stack = CUDA.zeros(Int32, num_examples(data), num_features(data)+3)
    @inbounds stack[:,1] .= 1 # start with 1 dec_id in the stack
    @inbounds stack[:,2] .= num_nodes(pbc) # start with the root in the stack
    num_threads = 256
    num_blocks = ceil(Int, size(state,1)/num_threads)
    CUDA.@sync begin
        @cuda threads=num_threads blocks=num_blocks map_cuda_kernel(num_leafs(pbc), params(pbc), nodes(pbc), elements(pbc), values, state, logprob, stack)
    end
    return state, logprob
end

function map_cuda_kernel(nl, params, nodes, elements, values, state, logprob, stack)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride_x = blockDim().x * gridDim().x
    for ex_id = index_x:stride_x:size(state,1)
        dec_id = pop_cuda!(stack, ex_id)
        while dec_id > zero(eltype(stack))
            if isleafgate(nl, dec_id)
                if isliteralgate(nl, dec_id)
                    l = literal(nl, dec_id)
                    var = lit2var(l)
                    @inbounds state[ex_id, var] = (l > 0) 
                end
            else
                edge_log_pr, prime, sub = map_child(params, nodes, elements, ex_id, dec_id, values)
                @inbounds logprob[ex_id] += edge_log_pr
                push_cuda!(stack, ex_id, prime)
                push_cuda!(stack, ex_id, sub)
            end
            dec_id = pop_cuda!(stack, ex_id)
        end
    end
    return nothing
end