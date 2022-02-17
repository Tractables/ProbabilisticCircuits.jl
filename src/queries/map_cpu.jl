
export MAP


"""
    MAP(pc::ProbCircuit, data::Matrix; batch_size, Float=Float32)

Evaluate max a posteriori (MAP) state of the circuit for given input(s) on cpu.

**Note**: This algorithm is only exact if the circuit is both decomposable and determinisitic.
If the circuit is only decomposable and not deterministic, this will give inexact results without guarantees.
"""
function MAP(pc::ProbCircuit, data::Matrix; batch_size, Float=Float32, return_map_prob=false)
    num_examples = size(data, 1)
    # log_likelihoods = zeros(Float32, num_examples)
    states = deepcopy(data)

    # Linearize PC
    linPC = linearize(pc)
    node2idx = Dict{ProbCircuit, UInt32}()
    for (i, node) in enumerate(linPC)
        node2idx[node] = i
    end

    nodes = size(linPC, 1)
    max_mars = zeros(Float, (batch_size, nodes))
    map_probs = zeros(Float, num_examples);

    for batch_start = 1:batch_size:num_examples
        batch_end = min(batch_start + batch_size - 1, num_examples)
        batch = batch_start:batch_end
        num_batch_examples = length(batch)

        max_mars .= zero(Float) # faster to zero out here rather than only in MulNodes
        eval_circuit_max!(max_mars, linPC, data, batch; node2idx, Float)
        
        map_probs[batch_start:batch_end] .= max_mars[1:num_batch_examples, end]    

        for (batch_idx, example_idx) in enumerate(batch)
            map_down_rec!(max_mars, pc, data, states, batch_idx, example_idx; node2idx, Float)
        end
    end
    if return_map_prob
        return states, map_probs
    else
        return states
    end
end

"""
    map_down_rec!(mars, node::ProbCircuit, data, states::Matrix, batch_idx, example_idx; node2idx::Dict{ProbCircuit, UInt32}, Float=Float32)

Downward pass on cpu for MAP. Recursively chooses the best (max) sum node children according to the "MAP upward pass" values.
Updates the missing values with map_state of that input node.
"""
function map_down_rec!(mars, node::ProbCircuit, data, states::Matrix, batch_idx, example_idx; 
    node2idx::Dict{ProbCircuit, UInt32}, Float=Float32)
    if isinput(node)
        if ismissing(data[example_idx, first(randvars(node))])
            states[example_idx, first(randvars(node))] = map_state(dist(node))
        end
    elseif ismul(node)
        for ch in inputs(node)
            map_down_rec!(mars, ch, data, states, batch_idx, example_idx; node2idx)
        end
    elseif issum(node)
        best_value = typemin(Float)
        best_child = nothing
        for (cidx, ch) in enumerate(inputs(node))
            child_mar_idx = node2idx[ch]
            val = mars[batch_idx, child_mar_idx] + node.params[cidx]
            if val > best_value
                best_value = val
                best_child = ch
            end
        end 
        map_down_rec!(mars, best_child, data, states, batch_idx, example_idx; node2idx)
    end
    return nothing
end


"""
    eval_circuit_max!(mars, linPC::AbstractVector{<:ProbCircuit}, data::Matrix, example_ids;  node2idx::Dict{ProbCircuit, UInt32}, Float=Float32)

Used internally. Evaluates the MAP upward pass of the circuit on cpu. Stores the values in `mars`.
- `mars`: (batch_size, nodes)
- `linPC`: linearized PC. (i.e. `linearize(pc)`)
- `data`: data Matrix (num_examples, features)
- `example_ids`: Array or collection of ids for current batch
- `node2idx`: Index of each ProbCircuit node in the linearized circuit
"""
function eval_circuit_max!(mars, linPC::AbstractVector{<:ProbCircuit}, data::Matrix, example_ids;  
    node2idx::Dict{ProbCircuit, UInt32}, Float=Float32)

    @inbounds for (mars_node_idx, node) in enumerate(linPC)
        if isinput(node)
            for (ind, example_idx) in enumerate(example_ids)
                mars[ind, mars_node_idx] =  if ismissing(data[example_idx, first(randvars(node))])
                    map_loglikelihood(dist(node))
                else
                    loglikelihood(dist(node), data[example_idx, first(randvars(node))])
                end
            end
        elseif ismul(node)        
            for ch in inputs(node)
                mars[:, mars_node_idx] .+= mars[:, node2idx[ch]]
            end
        elseif issum(node)
            mars[:, mars_node_idx] .= typemin(Float)
            for (cidx, ch) in enumerate(inputs(node))
                child_mar_idx = node2idx[ch]
                mars[:, mars_node_idx] .= max.(mars[:, mars_node_idx], mars[:, child_mar_idx] .+ node.params[cidx])
            end    
        end
    end
    return nothing
end