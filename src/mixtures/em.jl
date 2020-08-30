export one_step_em, component_weights_per_example, initial_weights, clustering,
log_likelihood_per_instance_per_component, estimate_parameters_cached, learn_em_model
using Statistics: mean
using LinearAlgebra: normalize!
using Clustering: kmeans, nclusters, assignments

function one_step_em(spc, train_x, component_weights;pseudocount)
    # E step
    lls = log_likelihood_per_instance_per_component(spc, train_x)
    lls .+= log.(component_weights)

    example_weights = component_weights_per_example(lls)
    component_weights .= sum(example_weights, dims=1)
    normalize!(component_weights, 1.0)

    # M step
    estimate_parameters_cached(spc, example_weights; pseudocount=pseudocount)
    logsumexp(lls, 2), component_weights
end

function component_weights_per_example(log_p_of_x_and_c)
    log_p_of_x = logsumexp(log_p_of_x_and_c, 2) # marginalize out components
    log_p_of_given_x_query_c = mapslices(col -> col .- log_p_of_x, log_p_of_x_and_c, dims=[1])
    p_of_given_x_query_c = exp.(log_p_of_given_x_query_c) # no more risk of underflow, so go to linear space
    @assert sum(p_of_given_x_query_c) ≈ size(log_p_of_x_and_c, 1) "$(sum(p_of_given_x_query_c)) != $(size(log_p_of_x_and_c))"# each row has proability 1
    Matrix(p_of_given_x_query_c)
end

function initial_weights(train_x, mix_num::Int64; alg="cluster")::Vector{Float64}
    if alg == "cluster"
        clustered = clustering(train_x, mix_num)
        counting = Float64.(num_examples.(clustered))
        return normalize!(counting, 1)
    elseif alg == "random"
        return normalize!(rand(Float64, mix_num), 1)
    else
        error("Initialize weights algorithm is $undefined")
    end
end

function clustering(train_x, mix_num::Int64; maxiter=200)::Vector
    train_x = Matrix(train_x)
    if mix_num == 1
        return [train_x]
    end
    
    n = num_examples(train_x)

    R = kmeans(train_x, mix_num; maxiter=maxiter)
    @assert nclusters(R) == mix_num
    a = assignments(R)

    clustered_train_x = Vector()
    for k in 1 : mix_num
        push!(clustered_train_x, train_x[:, findall(x -> x == k, a)]')
    end

    return clustered_train_x
end

function log_likelihood_per_instance_per_component(pc::SharedProbCircuit, data)
    @assert isbinarydata(data) "Can only calculate EVI on Bool data"
    
    compute_flows(pc, data)
    num_mix = num_components(pc)
    log_likelihoods = zeros(Float64, num_examples(data), num_mix)
    indices = init_array(Bool, num_examples(data))::BitVector
    
    
    ll(n::SharedProbCircuit) = ()
    ll(n::SharedProb⋁Node) = begin
        if num_children(n) != 1 # other nodes have no effect on likelihood
            for i in 1 : num_children(n)
                c = children(n)[i]
                log_theta = reshape(n.log_thetas[i, :], 1, num_mix)
                indices = get_downflow(n, c)
                view(log_likelihoods, indices::BitVector, :) .+=  log_theta # see MixedProductKernelBenchmark.jl
            end
         end
    end

    foreach(ll, pc)
    log_likelihoods
end

function estimate_parameters_cached(pc::SharedProbCircuit, example_weights; pseudocount::Float64)
    foreach(pc) do pn
        if is⋁gate(pn)
            if num_children(pn) == 1
                pn.log_thetas .= 0.0
            else
                smoothed_flow = Float64.(sum(example_weights[get_downflow(pn), :], dims=1)) .+ pseudocount
                uniform_pseudocount = pseudocount / num_children(pn)
                children_flows = vcat(map(c -> sum(example_weights[get_downflow(pn, c), :], dims=1), children(pn))...)
                @. pn.log_thetas = log((children_flows + uniform_pseudocount) / smoothed_flow)
                @assert all(sum(exp.(pn.log_thetas), dims=1) .≈ 1.0) "Parameters do not sum to one locally"
                # normalize away any leftover error
                pn.log_thetas .-= logsumexp(pn.log_thetas, dims=1)
            end
        end
    end
end

function learn_em_model(pc, train_x;
        num_mix=5,
        pseudocount=1.0,
        maxiter=typemax(Int))
    spc = SharedProbCircuit(pc, num_mix)
    compute_flows(spc, train_x)
    estimate_parameters_cached(spc, ones(Float64, num_examples(train_x), num_mix) ./ num_mix; pseudocount=pseudocount)
    component_weights = reshape(initial_weights(train_x, num_mix), 1, num_mix)

    for iter in 1 : maxiter
        @assert isapprox(sum(component_weights), 1.0; atol=1e-10)
        lls, component_weights = one_step_em(spc, train_x, component_weights; pseudocount=pseudocount)
        println("Log likelihood per instance is $(mean(lls))")
    end
end