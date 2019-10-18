import StatsBase

function bootstrap_samples_ids(train_x::PlainXData, n_samples::Int
                               #, rand_gen::AbstractRNG
                               )
    n_instances = num_examples(train_x)
    ids = 1:n_instances
    return [StatsBase.sample(
        #rand_gen,
        ids, n_instances, replace=true) for i in 1:n_samples]
end
    
function train_bagging(# pcs::Vector{<:ProbCircuit},
                        train_x::XBatches{Bool},
                        n_components::Int64;
                        init_models=nothing,
                        mixture_weights,
                        learn_base_estimator,
                        base_estimator_params,
                        logs)
                        
    @assert length(logs) == n_components "Dimension not match in train bagging."
    # bootstrapping samples
    bagging_samples = init_bagging_samples(train_x, n_components)
    
    # weights
    weights = nothing
    if mixture_weights == "uniform"
        weights = ones(Float64, n_components) ./ n_components
    else
        throw(DomainError(mixture_weights, "Unrecognized mixture weight mode"))
    end

    if issomething(init_models)
        @assert length(init_models) == n_components "Dimension not match in train bagging."
        for i in 1 : n_components
            learn_base_estimator(bagging_samples[i], init_models[i]; log=logs[i], base_estimator_params...)
        end
    
    else
        for i in 1 : n_components
            learn_base_estimator(bagging_samples[i]; log=logs[i], base_estimator_params...)
        end
    end
end

function init_bagging_samples(train_x::XBatches{Bool}, num_bags::Int64)::Vector{XBatches{Bool}}
    batch_size = max_batch_size(train_x)
    
    unbatched = unbatch(train_x)
    m = feature_matrix(unbatched)
    bagging_samples = Vector{XBatches{Bool}}()

    bootstrapped_ids = bootstrap_samples_ids(unbatched, num_bags)

    for i in 1 : num_bags
        new_examples = PlainXData(m[bootstrapped_ids[i], :])
        push!(bagging_samples, batch(new_examples, batch_size))
    end
    bagging_samples
end
