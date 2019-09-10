using StatsBase

function bootstrap_samples_ids(train_x::PlainXData, n_samples::Int
                               #, rand_gen::AbstractRNG
                               )
    n_instances = num_examples(train_x)
    ids = 1:n_instances
    return [StatsBase.sample(
        #rand_gen,
        ids, n_instances, replace=true) for i in 1:n_samples]
end

function learn_mixture_bagging2(train_x::PlainXData,
                               n_components::Int,
                               #, rand_gen::AbstractRNG,
                               learn_base_estimator;
                               base_estimator_params,
                               mixture_weights="uniform")
    #
    # bootstrapping samples
    bootstrapped_ids = bootstrap_samples_ids(train_x, n_components)
    n_components = length(bootstrapped_ids)
    

    weights = nothing
    if mixture_weights == "uniform"
        weights = [1/n_components for i in 1:n_components]
    else
        throw(DomainError(mixture_weights, "Unrecognized mixture weight mode"))
    end
    println(typeof(bootstrapped_ids[1]))
    components = [learn_base_estimator(PlainXData(train_x.x[b_ids,:]))
                  for b_ids in bootstrapped_ids]
        
    return components, weights
end
    
