export sample

import Random: default_rng


"""
    sample(pc::ProbCircuit, num_samples; rng = default_rng())

Generate `num_samples` from the joint distribution of the circuit without any conditions.
"""
function sample(pc::ProbCircuit, num_samples; batch_size, rng = default_rng(), Float=Float32)
    states, prs = sample(pc, num_samples, Matrix([missing for i=1:num_randvars(pc)]); batch_size, rng, Float)
end

"""
    sample(pc::ProbCircuit, num_samples; rng = default_rng())

Generate `num_samples` from the joint distribution of the circuit conditined on the `data`.
"""
function sample(pc::ProbCircuit, num_samples, data::Matrix; batch_size, rng = default_rng(), Float=Float32)
    num_examples = size(data, 1)

    # Linearize PC
    linPC = linearize(pc)
    node2idx = Dict{ProbCircuit, UInt32}()
    for (i, node) in enumerate(linPC)
        node2idx[node] = i
    end

    states = zeros(Union{Missing,eltype(data)}, num_samples, size(data, 1), size(data, 2))
    nodes = size(linPC, 1)
    values = zeros(Float, (batch_size, nodes))
    batch = 1:num_examples # do all in one batch for now

    eval_circuit!(values, linPC, data, batch; node2idx, Float)
    sample_down(pc, values, states, data, num_samples; rng, node2idx, Float)

    return states
end

function sample_down(pc::ProbCircuit, values, states, data, num_samples; rng, node2idx::Dict{ProbCircuit, UInt32}, Float)
    for (s_id, ex_id) = collect(Iterators.product(1:size(states,1), 1:size(states,2)))
        sample_rec!(pc, states, values, data; s_id, ex_id, rng, node2idx)
    end
    return nothing
end

function sample_rec!(node::ProbCircuit, states, values, data; s_id, ex_id, rng, node2idx::Dict{ProbCircuit, UInt32})
    if isinput(node)
        if ismissing(data[ex_id, first(randvars(node))])
            states[s_id, ex_id, first(randvars(node))] = sample_state(dist(node))
        else
            states[s_id, ex_id, first(randvars(node))] = data[ex_id, first(randvars(node))]
        end
    elseif ismul(node)
        for ch in inputs(node)
            sample_rec!(ch, states, values, data; s_id, ex_id, rng, node2idx)
        end
    elseif issum(node)
        sampled_child = inputs(node)[end]
        threshold = log(rand(rng)) + values[ex_id, node2idx[node]]
        cumul_prob = typemin(Float32)
        for (cid, ch) in enumerate(inputs(node))
            cumul_prob = logsumexp(cumul_prob, node.params[cid] + values[ex_id, node2idx[ch]])
            if cumul_prob > threshold
                sampled_child = ch
                break
            end
        end
        sample_rec!(sampled_child, states, values, data; s_id, ex_id, rng, node2idx)
    end
    return nothing
end