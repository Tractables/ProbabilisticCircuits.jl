export one_step_em, 
    component_weights_per_example, 
    initial_weights, 
    clustering,
    log_likelihood_per_instance_per_component, 
    estimate_parameters_cached, 
    learn_circuit_mixture, 
    learn_strudel

using Statistics: mean
using LinearAlgebra: normalize!
using Clustering: kmeans, nclusters, assignments
using DataFrames

function one_step_em(spc, data, values, flows, component_weights; pseudocount)
    # E step
    lls = log_likelihood_per_instance_per_component(spc, data, values, flows)
    lls .+= log.(component_weights)

    example_weights = component_weights_per_example(lls)
    component_weights .= sum(example_weights, dims=1)
    normalize!(component_weights, 1.0)

    # M step
    estimate_parameters_cached(spc, example_weights, values, flows; pseudocount=pseudocount)
    logsumexp(lls, 2), component_weights
end

function component_weights_per_example(log_p_of_x_and_c)
    log_p_of_x = logsumexp(log_p_of_x_and_c, 2) # marginalize out components
    log_p_of_given_x_query_c = mapslices(col -> col .- log_p_of_x, log_p_of_x_and_c, dims=[1])
    p_of_given_x_query_c = exp.(log_p_of_given_x_query_c) # no more risk of underflow, so go to linear space
    @assert sum(p_of_given_x_query_c) ≈ size(log_p_of_x_and_c, 1) "$(sum(p_of_given_x_query_c)) != $(size(log_p_of_x_and_c))"# each row has proability 1
    Matrix(p_of_given_x_query_c)
end

function initial_weights(data, mix_num::Int64; alg="cluster")::Vector{Float64}
    if alg == "cluster"
        clustered = clustering(data, mix_num)
        counting = Float64.(num_examples.(clustered))
        return normalize!(counting, 1)
    elseif alg == "random"
        return normalize!(rand(Float64, mix_num), 1)
    else
        error("Initialize weights algorithm is $undefined")
    end
end

function clustering(data, mix_num::Int64; maxiter=200)::Vector
    n = num_examples(data)
    if mix_num == 1
        return [data]
    end
    data = Matrix(data)
    R = kmeans(data, mix_num; maxiter=maxiter)
    @assert nclusters(R) == mix_num
    a = assignments(R)

    clustered_data = Vector()
    for k in 1 : mix_num
        push!(clustered_data, DataFrame(data[:, findall(x -> x == k, a)]'))
    end

    return clustered_data
end


function log_likelihood_per_instance_per_component(pc::SharedProbCircuit, data::DataFrame, values::Matrix{UInt64}, flows::Matrix{UInt64})
    @assert isbinarydata(data) "Can only calculate EVI on Bool data"
    
    N = num_examples(data)
    num_mix = num_components(pc)
    log_likelihoods = zeros(Float64, N, num_mix)
    indices = init_array(Bool, N)::BitVector
    
    
    ll(n::SharedProbCircuit) = ()
    ll(n::SharedSumNode) = begin
        if num_children(n) != 1 # other nodes have no effect on likelihood
            for i in 1 : num_children(n)
                c = children(n)[i]
                log_theta = reshape(n.log_probs[i, :], 1, num_mix)
                indices = downflow_all(values, flows, N, n, c)
                view(log_likelihoods, indices::BitVector, :) .+=  log_theta
            end
         end
    end

    foreach(ll, pc)
    log_likelihoods
end

function estimate_parameters_cached(pc::SharedProbCircuit, example_weights::Matrix{Float64}, 
        values::Matrix{UInt64}, flows::Matrix{UInt64}; pseudocount::Float64)
    N = size(example_weights, 1)
    foreach(pc) do pn
        if is⋁gate(pn)
            if num_children(pn) == 1
                pn.log_probs .= 0.0
            else
                smoothed_flow = Float64.(sum(example_weights[downflow_all(values, flows, N, pn), :], dims=1)) .+ pseudocount
                uniform_pseudocount = pseudocount / num_children(pn)
                children_flows = vcat(map(c -> sum(example_weights[downflow_all(values, flows, N, pn, c), :], dims=1), children(pn))...)
                @. pn.log_probs = log((children_flows + uniform_pseudocount) / smoothed_flow)
                @assert all(sum(exp.(pn.log_probs), dims=1) .≈ 1.0) "Parameters do not sum to one locally"
                # normalize away any leftover error
                pn.log_probs .-= logsumexp(pn.log_probs, dims=1)
            end
        end
    end
end


"""
Learns a mixture of circuits

    learn_strudel (train_x; init_maxiter = 10, em_maxiter=20)

See Strudel: Learning Structured-Decomposable Probabilistic Circuits. https://arxiv.org/abs/2007.09331
"""
function learn_strudel(train_x; num_mix = 5,
    pseudocount=1.0,
    init_maxiter = 10, 
    em_maxiter = 20)

    pc = learn_circuit(train_x, maxiter=init_maxiter)
    learn_circuit_mixture(pc, train_x; num_mix = num_mix, pseudocount= pseudocount, em_maxiter=em_maxiter)
end  


"""
Given a circuit, learns a mixture of structure decomposable circuits based on that circuit

    learn_circuit_mixture(pc, data; num_mix=5, pseudocount=1.0, maxiter=20)
"""
function learn_circuit_mixture(pc, data;
        num_mix=5,
        pseudocount=1.0,
        em_maxiter=20)

    spc = compile(SharedProbCircuit, pc, num_mix)
    values, flows = satisfies_flows(spc, data)
    component_weights = reshape(initial_weights(data, num_mix), 1, num_mix)
    estimate_parameters_cached(spc, ones(Float64, num_examples(data), num_mix) ./ num_mix, values, flows; pseudocount=pseudocount)

    lls = nothing
    for iter in 1 : em_maxiter
        @assert isapprox(sum(component_weights), 1.0; atol=1e-10)
        lls, component_weights = one_step_em(spc, data, values, flows, component_weights; pseudocount=pseudocount)
        println("EM Iteration $iter/$em_maxiter. Log likelihood $(mean(lls))")
    end
    spc, component_weights, lls
end


