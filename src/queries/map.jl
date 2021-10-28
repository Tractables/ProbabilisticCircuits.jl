export max_a_posteriori, MAP, map_prob

import DataFrames: DataFrame, mapcols!


#####################
# Circuit MAP/MPE evaluation
#####################

""" 
    max_a_posteriori(root::ProbCircuit, data::Union{Bool,Missing}...)
    max_a_posteriori(root::ProbCircuit, data::Union{Vector{<:Union{Bool,Missing}},CuVector{UInt8}})
    max_a_posteriori(circuit::ProbCircuit, data::DataFrame)
    max_a_posteriori(pbc::ParamBitCircuit, data; Float=Float32)

Evaluate maximum a-posteriori state of the circuit for given input(s).

Outputs the states, and the corresponding probabilities (in log domain).
"""
max_a_posteriori(root::ProbCircuit, data::Union{Bool,Missing}...) =
    max_a_posteriori(root, collect(Union{Bool,Missing}, data))

max_a_posteriori(root::ProbCircuit, data::Union{Vector{<:Union{Bool,Missing}},CuVector{UInt8}}) = begin
    map, prob = max_a_posteriori(root, DataFrame(reshape(data, 1, :), :auto))
    example(map, 1), prob[1]
end
    

max_a_posteriori(circuit::ProbCircuit, data::DataFrame; Float=Float32) =
    max_a_posteriori(same_device(ParamBitCircuit(circuit, data), data), data; Float)

function max_a_posteriori(pbc::ParamBitCircuit, data; Float=Float32)
    @assert isgpu(data) == isgpu(pbc) "ParamBitCircuit and data need to be on the same device"
    values = map_prob_all(pbc, data; Float)
    map_state =  map_down(pbc, data, values)
    @assert isgpu(values) == isgpu(map_state)
    map_state, values[:,end]
end

"""
Maximum a-posteriori queries
"""
const MAP = max_a_posteriori

"""
Mode of the distribution
"""
const mode = max_a_posteriori


#####################
# Circuit marginal evaluation
#####################

"""
    map_prob(root::ProbCircuit, data::Union{Real,Missing}...)
    map_prob(root::ProbCircuit, data::Union{Vector{Union{Bool,Missing}},CuVector{UInt8}})
    map_prob(circuit::ProbCircuit, data::DataFrame)
    map_prob(circuit::ParamBitCircuit, data::DataFrame)
    
The most likely world probability that agrees with the provided data"

Missing values should be denoted by `missing` in the data.
"""
map_prob(root::ProbCircuit, data::Union{Real,Missing}...) =
    map_prob(root, collect(Union{Bool,Missing}, data))

map_prob(root::ProbCircuit, data::Union{Vector{Union{Bool,Missing}},CuVector{UInt8}}) =
    map_prob(root, DataFrame(reshape(data, 1, :), :auto))[1]

map_prob(circuit::ProbCircuit, data::DataFrame) =
    map_prob(same_device(ParamBitCircuit(circuit, data), data) , data)

map_prob(circuit::ParamBitCircuit, data::DataFrame) =
    map_prob_all(circuit, data)[:,end]

#####################
# MAP Probability (upward pass)
#####################

"The most likely world that agrees with the provided data for all nodes"
map_prob_all(circuit::ProbCircuit, data::DataFrame) =
    map_prob_all(same_device(ParamBitCircuit(circuit, data), data), data)

function map_prob_all(circuit::ParamBitCircuit, data, reuse=nothing; Float=Float32)
    @assert num_features(data) == num_features(circuit) 
    # here we can reuse `init_marginal` from the marginal computation because 
    # our currently supported leafs all have MAP state `1`
    values = init_marginal(data, reuse, num_nodes(circuit); Float)
    map_prob_layers(circuit, values)
    return values
end

# upward pass helpers on CPU

"Compute MAP probabilities on the CPU (SIMD & multi-threaded)"
function map_prob_layers(circuit::ParamBitCircuit, values::Matrix)
    bc::BitCircuit = circuit.bitcircuit
    els = bc.elements
    pars = circuit.params
    for layer in bc.layers[2:end]
        Threads.@threads for dec_id in layer
            j = @inbounds bc.nodes[1,dec_id]
            els_end = @inbounds bc.nodes[2,dec_id]
            if j == els_end
                # first element value is just marginal value
                assign_marginal(values, dec_id, els[2,j], els[3,j], pars[j])
                j += 1
            else
                assign_map_prob(values, dec_id, els[2,j], els[3,j], els[2,j+1], els[3,j+1], pars[j], pars[j+1])
                j += 2
            end
            while j <= els_end
                accum_map_prob(values, dec_id, els[2,j], els[3,j], pars[j])
                j += 1
            end
        end
    end
end

assign_map_prob(v::Matrix{<:AbstractFloat}, i, e1p, e1s, e2p, e2s, p1, p2) = begin
    @avx for j=1:size(v,1)
        @inbounds x = v[j,e1p] + v[j,e1s] + p1
        @inbounds y = v[j,e2p] + v[j,e2s] + p2
        @inbounds v[j,i] = max(x, y)
    end
end

accum_map_prob(v::Matrix{<:AbstractFloat}, i, e1p, e1s, p1) = begin
    @avx for j=1:size(v,1)
        @inbounds x = v[j,i]
        @inbounds y = v[j,e1p] + v[j,e1s] + p1
        @inbounds v[j,i] = max(x, y)
    end
end

# upward pass helpers on GPU

"Compute marginals on the GPU"
function map_prob_layers(circuit::ParamBitCircuit, values::CuMatrix;  dec_per_thread = 8, log2_threads_per_block = 8)
    circuit = to_gpu(circuit)
    bc = circuit.bitcircuit
    CUDA.@sync for layer in bc.layers[2:end]
        num_examples = size(values, 1)
        num_decision_sets = length(layer)/dec_per_thread

        kernel = @cuda name="map_prob_layers_cuda" launch=false map_prob_layers_cuda(layer, bc.nodes, bc.elements, circuit.params, values)
        config = launch_configuration(kernel.fun)
        threads, blocks = balance_threads_2d(num_examples, num_decision_sets, config.threads)
        kernel(layer, bc.nodes, bc.elements, circuit.params, values
            ; threads, blocks)
    end
end

"CUDA kernel for circuit evaluation"
function map_prob_layers_cuda(layer, nodes, elements, params, values)
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
                x = max(x, y)
            end
            values[j, decision_id] = x
        end
    end
    return nothing
end

#####################
# MAP state (downward pass)
#####################

# helper function

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
    @inbounds return elements[2,j_opt], elements[3,j_opt] 
end

# CPU code

function map_down(pbc, data, values::Array)
    state = zeros(Bool, num_examples(data), num_features(data))
    Threads.@threads for ex_id = 1:size(state,1)
        map_rec(num_leafs(pbc), params(pbc), nodes(pbc), elements(pbc), ex_id, num_nodes(pbc), values, state)
    end
    df = DataFrame(state, :auto)
    mapcols!(c -> BitVector(c), df)
    return df
end

function map_rec(nl, params, nodes, elements, ex_id, dec_id, values, state)
    if isleafgate(nl, dec_id)
        if isliteralgate(nl, dec_id)
            l = literal(nl, dec_id)
            @inbounds state[ex_id, lit2var(l)] = (l > 0) 
        end
    else
        prime, sub = map_child(params, nodes, elements, ex_id, dec_id, values)
        map_rec(nl, params, nodes, elements, ex_id, prime, values, state)
        map_rec(nl, params, nodes, elements, ex_id, sub,   values, state)
    end
end

# GPU code

function map_down(pbc, data, values::CuArray)
    state = CUDA.zeros(Bool, num_examples(data), num_features(data))
    stack = CUDA.zeros(Int32, num_examples(data), num_features(data)+3)
    @inbounds stack[:,1] .= 1 # start with 1 dec_id in the stack
    @inbounds stack[:,2] .= num_nodes(pbc) # start with the root in the stack
    
    CUDA.@sync begin
        kernel = @cuda name="map_cuda_kernel" launch=false map_cuda_kernel(num_leafs(pbc), params(pbc), nodes(pbc), elements(pbc), values, state, stack)
        config = launch_configuration(kernel.fun)
        threads = config.threads
        blocks = cld(size(state,1), threads)
        kernel(num_leafs(pbc), params(pbc), nodes(pbc), elements(pbc), values, state, stack
            ; threads, blocks)
    end
    # do the conversion to a CuBitVector on the CPU...
    df = DataFrame(to_cpu(state), :auto)
    mapcols!(c -> to_gpu(BitVector(c)), df)
    return df
end

function map_cuda_kernel(nl, params, nodes, elements, values, state, stack)
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
                prime, sub = map_child(params, nodes, elements, ex_id, dec_id, values)
                push_cuda!(stack, prime, ex_id)
                push_cuda!(stack, sub, ex_id)
            end
            dec_id = pop_cuda!(stack, ex_id)
        end
    end
    return nothing
end