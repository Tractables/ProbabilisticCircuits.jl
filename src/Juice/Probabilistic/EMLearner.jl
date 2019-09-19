"""
Train a mixture of probabilistic circuits from data, starting with random example weights.
"""
function train_mixture( pcs::Vector{<:ProbCircuit△},
                        train_x::XBatches{Bool},
                        pseudocount, iters,
                        structure_learner=nothing, learnstruct_step = iters + 1, # structure learning
                        logger=nothing, logging_step = 1 # logging or saving results
                      )::AbstractFlatMixture

    # create mixture model with arbitrary component weights
    mixture = FlatMixture(pcs)

    # create random example weight matrix
    example_weights = random_example_weights(num_components(afcs), num_examples(train_x))

    train_mixture(mixture, train_x, pseudocount, iters, example_weights, 
                  structure_learner, learnstruct_step, logger, logging_step)
end

"""
Train a mixture model from data.
Learning is initialized from either
- the parameters stored in the given mixture, if no `example_weights` are given
- example weights, if they are given in `example_weights`
When a `structure_learner` is given, it will be called between the E and M step to update circuit structures.
"""
function train_mixture( mixture::AbstractFlatMixture, # we start from component weights that are already given
                        train_x::XBatches{Bool},
                        pseudocount, num_iters,
                        example_weights::Union{Nothing, Matrix}=nothing,
                        structure_learner=nothing, learnstruct_step = num_iters + 1, # structure learning
                        logger=nothing, logging_step = 1 # logging or saving results
                      )::AbstractFlatMixture
    
    @assert feature_type(train_x) == Bool "Can only learn probabilistic circuits on Bool data"

    # initialize data structures
    mixture_flow, aggr_circuits, flow_circuits = init_auxiliary_circuits(mixture, train_x)

    # if we are starting from the existing mixture parameters, we are good to go for the first expectation step
    # otherwise we need to estimate the parameters from the given `example_weights` first in a 0th maximization step
    start_iter = issomething(example_weights) ? 0 : 1
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #TODO: finish these refactorings!!!!!!!!!!!!!!!!!!!!!!
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for i in start_iter:num_iters

        # Note: for now we are collecting all example weights before re-estimating the parameters. 
        # If memory usage becomes and issue because of large datasets with many mixture components, we can interleave the E and M step for each batch. 

        # Expectation step (update example weights given mixture parameters)
        if i > 0
            aggr_component_probability, example_weights = expectation_step(mixture_flow, aggr_circuits, flow_circuits, train_x)
        end

        # Optional structure refinement step

        # Maximization step (update mixture parameters given example weights)
        if i % step == 0 && issomething(structure_learner)
            component_weights .= aggr_component_probability ./ sum(aggr_component_probability)
            pcs = structure_learner(train_x, pcs, example_weights)
            afcs, fcs_parameters, fcs_weights, component_weights = initial_mixture_trainer(pcs, train_x, component_weights; pseudocount=pseudocount, example_weights = example_weights)
        
        else
            maximization_step(aggr_component_probability, afcs, component_weights; pseudocount=pseudocount)
        end

        if i % log_step == 0 && issomething(logger)
            logger(mixture_flow, train_x)
        end
    end

end

function init_auxiliary_circuits(mixture::AbstractFlatMixture, train_x::XBatches{Bool})
    mixture_flow = ensure_with_flows(mixture, max_batch_size(train_x))    
    aggr_circuits = [AggregateFlowCircuit(pc, Float64) for pc in components(mixture)]
    flow_circuits = [FlowCircuit(afc, max_batch_size(train_x), Bool) for afc in aggr_circuits]
    return mixture_flow, aggr_circuits, flow_circuits
end

function expectation_step(mixture_flow::FlatMixtureWithFlow, train_x::XBatches{Bool})::Vector{Matrix{Float64}}
    map(log_likelihood_per_instance_component(mixture_flow, train_x)) do log_p_of_x_and_c
        log_p_of_x = logsumexp(log_p_of_x_and,2)
        log_p_of_given_x_query_c = mapslices(xcol -> xrow .- log_p_of_x, log_p_of_x_and_c, dims=[1])
        log_p_of_given_x_query_c # these are the component weights for each example
    end
end

function estimate_parameters(pcs, train_x, w; pseudocount, data_weights)
    # initialize data structures
    afcs = [AggregateFlowCircuit(pc, Float64) for pc in pcs]
    fcs_parameters = [FlowCircuit(afc, max_batch_size(train_x), Bool, FlowCache(), opts_accumulate_flows) for afc in afcs]
    fcs_weights = [FlowCircuit(pc, max_batch_size(train_x), Bool, FlowCache(), opts_accumulate_flows) for pc in pcs]
    component_weights = issomething(w) ? copy(w) :  ones(Float64, num_components(pcs)) ./ num_components(pcs)

    # reset aggregate flows
    foreach(afc->reset_aggregate_flows(afc), afcs)

    # learn initial models

    # Aggregating initial flows
    weights_gen = issomething(w) ? ones : rand
    initial_mixture_model(fcs_parameters, train_x; weights_gen=weights_gen, data_weights=data_weights)

    # Estimating initial parameters
    estimate_parameters_from_aggregates(afcs; pseudocount=pseudocount)

    return afcs, fcs_parameters, fcs_weights, component_weights
end

function initial_mixture_model(fcs_parameters, train_x; weights_gen, data_weights::Union{Nothing, Matrix})
    sum_n = 0
    for batch in train_x
        sum_n += n = num_examples(batch)
        # accumulate flows for random weights to initialize parameters
        for (i, fc) in enumerate(fcs_parameters)
            if issomething(data_weights)
                wbatch = WXData(batch, data_weights[sum_n - n + 1: sum_n, i])
            else
                wbatch = WXData(batch, weights_gen(Float32, num_examples(batch)))
            end
            accumulate_aggr_flows_batch(fc, wbatch)
        end
    end
end

function maximization_step(aggr_component_probability, afcs, component_weights; pseudocount)
    component_weights .= aggr_component_probability ./ sum(aggr_component_probability)
    for afc in afcs
        estimate_parameters(afc; pseudocount=pseudocount)
    end
end


"Create random example weights that sum to one overall components"
function random_example_weights(num_components, num_examples)
    w = rand(Float64, num_components, num_examples)
    w ./ sum(w;dims=2)
end