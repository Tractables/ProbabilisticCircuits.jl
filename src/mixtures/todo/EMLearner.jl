"""
Train a mixture of probabilistic circuits from data, starting with random example weights.
"""
function train_mixture( pcs::Vector{<:ProbCircuit},
                        train_x::XBatches{Bool},
                        pseudocount, num_iters;
                        structure_learner=nothing, learnstruct_step = num_iters + 1, # structure learning
                        logger=nothing, logging_step = 1 # logging or saving results
                      )::AbstractFlatMixture


    # create mixture model with uniform component weights
    mixture_flow = init_mixture_with_flows(FlatMixture(pcs), train_x)

    # reset aggregate statistics
    reset_mixture_aggregate_flows(mixture_flow)

    # do a quick maximization step
    for batch in train_x
        example_weights = random_example_weights(num_examples(batch), num_components(mixture_flow))
        aggregate_flows(mixture_flow, batch, example_weights)
    end
    estimate_parameters(mixture_flow, component_weights(mixture_flow); pseudocount=pseudocount)

    train_mixture(mixture_flow, train_x, pseudocount, num_iters; 
                structure_learner=structure_learner, learnstruct_step=learnstruct_step, 
                logger=logger, logging_step=logging_step)
end

"""
Train a mixture model from data.
Learning is initialized from the parameters stored in the given mixture.
When a `structure_learner` is given, it will be called between EM steps to update circuit structures.
"""
function train_mixture( mixture::AbstractFlatMixture, # we start from component weights that are already given
                        train_x::XBatches{Bool},
                        pseudocount, num_iters;
                        structure_learner=nothing, learnstruct_step = num_iters + 1, # structure learning
                        logger=nothing, logging_step = 1 # logging or saving results
                      )::AbstractFlatMixture
    
    @assert feature_type(train_x) == Bool "Can only learn probabilistic circuits on Bool data"

    # initialize data structures
    mixture_flow = init_mixture_with_flows(mixture, train_x)

    if issomething(logger)
        logger(mixture_flow)
    end
    
    for i in 1:num_iters

        # reset aggregate statistics
        total_component_probability = ones(Float64, num_components(mixture_flow)) .* pseudocount ./ num_components(mixture_flow)
        reset_mixture_aggregate_flows(mixture_flow)

        # are we doing structure learning at the end of this iteration?
        is_learnstruct_iter = issomething(structure_learner) && i % learnstruct_step == 0 

        all_example_weights = Vector{Matrix{Float64}}()

        # Expectation step (update example weights given mixture parameters)
        # + collecting aggregate statistics for subsequent maximization step
        for batch in train_x
            log_p_of_x_and_c = log_likelihood_per_instance_component(mixture_flow, batch)
            example_weights = component_weights_per_example(log_p_of_x_and_c)
            
            # copy the flows already computed by `log_likelihood_per_instance_component` into the underlying aggregate flow circuit
            # this way the maximization step can use them to estimate new parameters
            aggregate_flows_cached(mixture_flow, batch, example_weights)

            # store the aggregated component probabilities such that the maximization step can re-estimate the component weights
            total_component_probability .+= dropdims(sum(example_weights, dims=1), dims=1)

            # cache the example weights for the structure learner at the end of this EM iteration
            is_learnstruct_iter && push!(all_example_weights, example_weights)
        end

        # Maximization step (update mixture parameters given example weights (as stored in aggregate circuits))
        estimate_parameters(mixture_flow, total_component_probability; pseudocount=pseudocount)

        # Structural EM step
        if is_learnstruct_iter
            new_mixture_flow = structure_learner(mixture_flow, train_x, all_example_weights)
            # mixture = replace_prob_circuits(mixture, new_pcs)
            # re-initialize data structures
            mixture_flow = init_mixture_with_flows(new_mixture_flow, train_x)
        end

        if i % logging_step == 0 && issomething(logger)
            logger(mixture_flow)
        end
    end

    return mixture_flow
end

"Ensure we have a FlatMixtureWithFlow where the flow circuits have aggregate flow circuits as origin"
function init_mixture_with_flows(mixture::FlatMixtureWithFlow, ::XBatches{Bool})::FlatMixtureWithFlow 
    if ! all(fc -> grand_origin(fc) isa AggregateFlowΔ, mixture.flowcircuits)
        init_mixture_with_flows(origin(mixture))
    else
        mixture 
    end
end
function init_mixture_with_flows(mixture::FlatMixture, train_x::XBatches{Bool})::FlatMixtureWithFlow
    aggr_circuits = [AggregateFlowΔ(pc, Float64) for pc in components(mixture)]
    flow_circuits = [FlowΔ(afc, max_batch_size(train_x), Bool, opts_accumulate_flows) for afc in aggr_circuits]
    FlatMixtureWithFlow(mixture, flow_circuits)
end

function reset_mixture_aggregate_flows(mixture_flow::FlatMixtureWithFlow)
    for fc in mixture_flow.flowcircuits
        reset_aggregate_flows(grand_origin(fc))
    end
end

"Compute the component weights for each example from likelihoods"
function component_weights_per_example(log_p_of_x_and_c)
    log_p_of_x = logaddexp(log_p_of_x_and_c, 2) # marginalize out components
    log_p_of_given_x_query_c = mapslices(col -> col .- log_p_of_x, log_p_of_x_and_c, dims=[1])
    p_of_given_x_query_c = exp.(log_p_of_given_x_query_c) # no more risk of underflow, so go to linear space
    @assert sum(p_of_given_x_query_c) ≈ size(log_p_of_x_and_c, 1) # each row has proability 1
    p_of_given_x_query_c
end

"Compute and aggregate flows for mixture components"
function aggregate_flows(mixture_flow, batch, example_weights)
    for i in 1:num_components(mixture_flow)
        fc =  mixture_flow.flowcircuits[i]
        wbatch = weighted_batch_for_component(batch, example_weights,i)
        accumulate_aggr_flows_batch(fc, wbatch)
    end
end

"Aggregate already-computed flows for mixture components"
function aggregate_flows_cached(mixture_flow, batch, example_weights)
    for i in 1:num_components(mixture_flow)
        fc =  mixture_flow.flowcircuits[i]
        wbatch = weighted_batch_for_component(batch, example_weights,i)
        accumulate_aggr_flows_cached(fc, wbatch)
    end
end

function estimate_parameters(mixture_flow, total_component_probability; pseudocount)
    component_weights(mixture_flow) .= total_component_probability ./ sum(total_component_probability)
    for fc in mixture_flow.flowcircuits
        estimate_parameters_cached(grand_origin(fc); pseudocount=pseudocount)
    end
end

"Get a new weighted batch for this component"
weighted_batch_for_component(batch::PlainXData, example_weights, component_i)::WXData =
    WXData(batch, example_weights[:,component_i])

"Create random example weights that sum to one overall components"
function random_example_weights(num_examples::Int, num_components::Int)::Matrix{Float64}
    w = rand(Float64, num_examples, num_components)
    w ./ sum(w;dims=2)
end