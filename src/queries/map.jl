export map_state, MAP

import DataFrames: DataFrame, mapcols!

#####################
# Circuit MAP/MPE evaluation
#####################

"Evaluate maximum a-posteriori state of the circuit for a given input"
map_state(root::ProbCircuit, data::Union{Bool,Missing}...) =
    map_state(root, collect(Union{Bool,Missing}, data))

map_state(root::ProbCircuit, data::Union{Vector{Union{Bool,Missing}},CuVector{UInt8}}) =
    map_state(root, DataFrame(reshape(data, 1, :)))[1,:]

map_state(circuit::ProbCircuit, data::DataFrame) =
    map_state(same_device(ParamBitCircuit(circuit, data), data), data)

function map_state(pbc::ParamBitCircuit, data; Float=Float32)
    @assert isgpu(data) == isgpu(pbc) "ParamBitCircuit and data need to be on the same device"
    values = marginal_all(pbc, data)
    if !isgpu(data)
        state = zeros(Bool, num_examples(data), num_features(data))
        logprob = zeros(Float, num_examples(data))
        map_state_rec_cpu(num_leafs(pbc), params(pbc), nodes(pbc), elements(pbc), num_nodes(pbc), values, state, logprob)
    else
        state = CUDA.zeros(Bool, num_examples(data), num_features(data))
        logprob = CUDA.zeros(Float, num_examples(data))
        stack = CUDA.zeros(Int32, num_examples(data), num_features(data)+3)
        stack[:,1] .= 1 # start with 1 dec_id in the stack
        stack[:,2] .= num_nodes(pbc) # start with the root in the stack
        num_threads = 256
        num_blocks = ceil(Int, size(state,1)/num_threads)
        CUDA.@sync begin
            @cuda threads=num_threads blocks=num_blocks map_state_iter_cuda(num_leafs(pbc), params(pbc), nodes(pbc), elements(pbc), values, state, logprob, stack)
        end
    end
    df = DataFrame(to_cpu(state))
    mapcols!(c -> BitVector(c), df)
    df, logprob
end

"""
Maximum a-posteriori queries
"""
const MAP = map_state

"Find the MAP child value and node id of a given decision node"
function map_child(params, nodes, elements, ex_id, dec_id, values)
    els_start = nodes[1,dec_id]
    els_end = nodes[2,dec_id]
    pr_opt = typemin(eltype(values))
    j_opt = 1
    for j = els_start:els_end
        prime = elements[2,j]
        sub = elements[3,j]
        pr = values[ex_id, prime] + values[ex_id, sub] + params[j]
        if pr > pr_opt
            pr_opt = pr
            j_opt = j
        end
    end
    return params[j_opt], elements[2,j_opt], elements[3,j_opt] 
end

# CPU code

function map_state_rec_cpu(nl, params, nodes, elements, dec_id, values, state, logprob)
    Threads.@threads for ex_id = 1:size(state,1)
        map_state_rec(nl, params, nodes, elements, ex_id, dec_id, values, state, logprob)
    end
end

function map_state_rec(nl, params, nodes, elements, ex_id, dec_id, values, state, logprob)
    if isleafgate(nl, dec_id)
        if isliteralgate(nl, dec_id)
            l = literal(nl, dec_id)
            state[ex_id, lit2var(l)] = (l > 0) 
        end
    else
        edge_log_pr, prime, sub = map_child(params, nodes, elements, ex_id, dec_id, values)
        logprob[ex_id] += edge_log_pr
        map_state_rec(nl, params, nodes, elements, ex_id, prime, values, state, logprob)
        map_state_rec(nl, params, nodes, elements, ex_id, sub,   values, state, logprob)
    end
end

# GPU code

function map_state_iter_cuda(nl, params, nodes, elements, values, state, logprob, stack)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride_x = blockDim().x * gridDim().x
    for ex_id = index_x:stride_x:size(state,1)
        dec_id = pop_cuda!(stack, ex_id)
        while dec_id > zero(eltype(stack))
            if isleafgate(nl, dec_id)
                if isliteralgate(nl, dec_id)
                    l = literal(nl, dec_id)
                    var = lit2var(l)
                    state[ex_id, Int(var)] = (l > 0) 
                end
            else
                edge_log_pr, prime, sub = map_child(params, nodes, elements, ex_id, dec_id, values)
                logprob[ex_id] += edge_log_pr
                push_cuda!(stack, ex_id, prime)
                push_cuda!(stack, ex_id, sub)
            end
            dec_id = pop_cuda!(stack, ex_id)
        end
    end
    return nothing
end

#TODO move to utils

function pop_cuda!(stack, i)
    if stack[i,1] == zero(eltype(stack))
        return zero(eltype(stack))
    else
        stack[i,1] -= one(eltype(stack))
        return stack[i,stack[i,1]+2]
    end
end

function push_cuda!(stack, i, v)
    stack[i,1] += one(eltype(stack))
    CUDA.@cuassert 1+stack[i,1] <= size(stack,2) "CUDA stack overflow"
    stack[i,1+stack[i,1]] = v
    return nothing
end

length_cuda(stack, i) = stack[i,1]
