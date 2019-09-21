#####################
# Probabilistic circuit mixtures
#####################

"A probabilistic mixture model"
abstract type AbstractMixture end

"A probabilistic mixture model whose components are not themselves mixtures"
abstract type AbstractFlatMixture <: AbstractMixture end

"A probabilistic mixture model of probabilistic circuits"
struct FlatMixture <: AbstractFlatMixture
    weights::Vector{Float64}
    components::Vector{<:ProbCircuit△}
    FlatMixture(w,c) = begin
        @assert length(w) == length(c)
        @assert sum(w) ≈ 1.0
        new(w,c)
    end
end

FlatMixture(c) = FlatMixture(uniform(length(c)),c)

"A mixture with cached flow circuits for each component (which are assumed to be ProbCircuits)"
struct FlatMixtureWithFlow <: AbstractFlatMixture
    origin::FlatMixture
    flowcircuits::Vector{<:FlowCircuit△}
    FlatMixtureWithFlow(origin,fcs) = begin
        @assert num_components(origin) == length(fcs)
        foreach(components(origin), fcs) do or, fc
            @assert or[end] === prob_origin(fc)[end]
        end
        new(origin,fcs)
    end
end

FlatMixtureWithFlow(w,c,f) = FlatMixtureWithFlow(FlatMixture(w,c),f)

#####################
# Functions
#####################

"Get the components in this mixture"
@inline components(m::FlatMixture) = m.components
@inline components(m::FlatMixtureWithFlow) = components(m.origin)

"Get the component weights in this mixture"
@inline component_weights(m::FlatMixture) = m.weights
@inline component_weights(m::FlatMixtureWithFlow) = component_weights(m.origin)

"Number of components in a mixture"
@inline num_components(m)::Int = length(components(m))

"Convert a given flat mixture into one with cached flows"
ensure_with_flows(m::FlatMixture, size_hint::Int)::FlatMixtureWithFlow = begin
    flowcircuits = [FlowCircuit(pc, size_hint, Bool, FlowCache(), opts_accumulate_flows) for pc in components(m)]
    FlatMixtureWithFlow(m,flowcircuits)
end
ensure_with_flows(m::FlatMixtureWithFlow, ::Int)::FlatMixtureWithFlow = m

replace_prob_circuits(m::FlatMixture, pcs::Vector{ProbCircuit△}) =
    FlatMixture(component_weights(m), pcs)

# log_likelihood

function log_likelihood(mixture::FlatMixture, batches::XBatches{Bool})::Float64
    mwf = ensure_with_flows(mixture, max_batch_size(batches))
    log_likelihood(mwf, batches)
end

function log_likelihood(mixture::FlatMixtureWithFlow, batches::XBatches{Bool})::Float64
    # assume the per-batch call will compute a weighted sum over examples
    sum(batch -> log_likelihood(mixture, batch), batches)
end

function log_likelihood(mixture::FlatMixtureWithFlow, batch::PlainXData{Bool})::Float64
    sum(log_likelihood_per_instance(mixture, batch))
end

# log_likelihood_per_instance (including mixture weight likelihood)

function log_likelihood_per_instance(mixture::FlatMixture, batches::XBatches{Bool})::Vector{Float64}
    mwf = ensure_with_flows(mixture, max_batch_size(batches))
    log_likelihood_per_instance(mwf, batches)
end

function log_likelihood_per_instance(mixture::FlatMixtureWithFlow, batches::XBatches{Bool})::Vector{Float64}
    mapreduce(b -> log_likelihood_per_instance(mixture, b), vcat, batches)
end

function log_likelihood_per_instance(mixture::FlatMixtureWithFlow, batch::PlainXData{Bool})::Vector{Float64}
    log_p_of_x_and_c = log_likelihood_per_instance_component(mixture, batch)
    logsumexp(log_p_of_x_and_c, 2)
end

# Log likelihoods per instance and component (including mixture weight likelihood)


"Log likelihood per instance and component. A vector of matrices per batch where the first dimension is instance, second is components."
function log_likelihood_per_instance_component(mixture::FlatMixtureWithFlow, batches::XBatches{Bool})::Vector{Matrix{Float64}}
    [log_likelihood_per_instance_component(mixture, batch) for batch in batches]
end

"Log likelihood per instance and component. First dimension is instance, second is components."
function log_likelihood_per_instance_component(mixture::FlatMixtureWithFlow, batch::PlainXData{Bool})::Matrix{Float64}
    hcat(log_likelihood_per_component_instance(mixture, batch)...)
end

"Log likelihood per component and instance. Outer vector is components, inner vector is instances"
function log_likelihood_per_component_instance(mixture::FlatMixtureWithFlow, batch::PlainXData{Bool})::Vector{Vector{Float64}}
    map(mixture.flowcircuits, component_weights(mixture)) do fc, component_weight
        log_likelihood_per_instance(fc, batch) .+ log(component_weight)
    end
end
