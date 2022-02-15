using Random

export loglikelihoods_cpu, loglikelihood_cpu

function loglikelihoods(pc::ProbCircuit, data::Matrix)
    num_examples = size(data, 1)
    log_likelihoods = zeros(Float32, num_examples)

    for idx = 1 : num_examples
        log_likelihoods[idx] = loglikelihood(pc, data, idx)
    end

    return log_likelihoods
end


function loglikelihood(root::ProbCircuit, data::Matrix, example_id; Float=Float32)
    f_i(node) = begin
        val = data[example_id, first(randvars(node))]
        ismissing(val) ? Float(0.0) : loglikelihood(dist(node), val)
    end
    f_m(node, ins) = begin
        sum(ins)
    end
    f_s(node, ins) = begin
        reduce(logsumexp, node.params .+ ins)  
    end
    foldup_aggregate(root, f_i, f_m, f_s, Float)
end