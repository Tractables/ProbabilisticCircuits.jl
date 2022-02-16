using Random

export loglikelihoods, loglikelihood, 
    loglikelihoods_vectorized, 
    loglikelihoods_linearized

"""
    loglikelihoods(pc::ProbCircuit, data::Matrix)

Computes loglikelihoods of the `circuit` over the `data` on cpu. Linearizes the circuit and computes the marginals in batches.
"""
function loglikelihoods(pc::ProbCircuit, data::Matrix; batch_size, Float=Float32)
    num_examples = size(data, 1)
    log_likelihoods = zeros(Float32, num_examples)

    # Linearize PC
    linPC = linearize(pc)
    node2idx = Dict{ProbCircuit, UInt32}()
    for (i, node) in enumerate(linPC)
        node2idx[node] = i
    end

    nodes = size(linPC, 1)
    mars = zeros(Float, (batch_size, nodes))

    for batch_start = 1:batch_size:num_examples
        batch_end = min(batch_start + batch_size - 1, num_examples)
        batch = batch_start:batch_end
        num_batch_examples = length(batch)
        
        eval_circuit!(mars, linPC, data, batch; node2idx, Float)
        log_likelihoods[batch_start:batch_end] .= mars[1:num_batch_examples, end]
        mars .= zero(Float) # faster to zero out here rather than only in MulNodes
    end
    return log_likelihoods
end

"""
    eval_circuit!(mars, linPC::AbstractVector{<:ProbCircuit}, data::Matrix, example_ids;  node2idx::Dict{ProbCircuit, UInt32}, Float=Float32)

Used internally. Evaluates the marginals of the circuit on cpu. Stores the values in `mars`.
- `mars`: (batch_size, nodes)
- `linPC`: linearized PC. (i.e. `linearize(pc)`)
- `data`: data Matrix (num_examples, features)
- `example_ids`: Array or collection of ids for current batch
- `node2idx`: Index of each ProbCircuit node in the linearized circuit
"""
function eval_circuit!(mars, linPC::AbstractVector{<:ProbCircuit}, data::Matrix, example_ids;  
       node2idx::Dict{ProbCircuit, UInt32}, Float=Float32)

    @inbounds for (mars_node_idx, node) in enumerate(linPC)
        if isinput(node)
            for (ind, example_idx) in enumerate(example_ids)
                mars[ind, mars_node_idx] = ismissing(data[example_idx, first(randvars(node))]) ? zero(Float) : loglikelihood(dist(node), data[example_idx, first(randvars(node))])
            end
        elseif ismul(node)        
            for ch in inputs(node)
                mars[:, mars_node_idx] .+= @view mars[:, node2idx[ch]]
            end
        else 
            @assert issum(node)
            mars[:, mars_node_idx] .= typemin(Float)
            for (cidx, ch) in enumerate(inputs(node))
                child_mar_idx = node2idx[ch]
                mars[:, mars_node_idx] .= logsumexp.(mars[:, mars_node_idx], mars[:, child_mar_idx] .+ node.params[cidx])
            end    
        end
    end
    return nothing
end

"""
    loglikelihood(root::ProbCircuit, data::Matrix, example_id; Float=Float32)

Computes marginal loglikelihood recursively on cpu for a single instance `data[example_id, :]`.

**Note**: Quite slow, only use for demonstration/educational purposes. 
"""
function loglikelihood(root::ProbCircuit, data::Matrix, example_id; Float=Float32)
    f_i(node) = begin
        val = data[example_id, first(randvars(node))]
        ismissing(val) ? Float(0.0) : loglikelihood(dist(node), val)
    end
    f_m(node, ins) = sum(ins)
    f_s(node, ins) = reduce(logsumexp, node.params .+ ins)  
    foldup_aggregate(root, f_i, f_m, f_s, Float)
end

"""
**Note**: Experimental**; will be removed or renamed later
"""
function loglikelihoods_vectorized(root::ProbCircuit, data::Matrix; Float=Float32)
    function logsumexp(vals::Vector{Float32})
        reduce(logsumexp, vals)
    end

    f_i(node) = begin
        [ismissing(data[idx, first(randvars(node))]) ? Float(0.0) : loglikelihood(dist(node), data[idx, first(randvars(node))]) for idx=1:size(data,1)]    
    end
    f_m(node, ins) = begin
        sum(ins)
    end
    f_s(node, ins) = begin        
        entry(i, data_idx) = node.params[i] + ins[i][data_idx]

        ans = zeros(Float, size(data, 1))
        for idx = 1:size(data, 1)
            ans[idx] = logsumexp([entry(i, idx) for i=1:size(node.params, 1)])
        end        
        ans
    end
    foldup_aggregate(root, f_i, f_m, f_s, Vector{Float})
end