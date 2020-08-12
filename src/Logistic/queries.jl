export class_weights_per_instance, class_likelihood_per_instance, downflow, accuracy, predict_class, 
    do_nothing_test, access_flow_test, dummy_calculation_test, dummy_calculation_test2

using LogicCircuits: UpDownFlow1, UpDownFlow2, or_nodes
using ..Probabilistic: get_downflow, get_upflow
using LoopVectorization: @avx, vifelse


"""
Class Conditional Probability
"""
# with flows computed (2.03 s (80704 allocations: 3.32 MiB))
# 5.136 s (275778 allocations: 6.99 GiB) on mnist.circuit
@inline function class_likelihood_per_instance(lc::LogisticCircuit,  classes::Int, data; flows_computed=false)
    if !flows_computed
        compute_flows(lc, data)
    end
    
    weights = class_weights_per_instance(lc, classes, data; flows_computed=true)
    @avx @. 1.0 / (1.0 + exp(-weights))
end

@inline function class_weights_per_instance(lc::LogisticCircuit,  classes::Int, data; flows_computed=false)
    if !flows_computed
        compute_flows(lc, data)
    end
    
    weights = zeros(num_examples(data), classes)
    foreach(or_nodes(lc)) do ln
        foreach(eachrow(ln.thetas), children(ln)) do theta, c
            flow = Float32.(downflow(ln, c))
            @avx @. weights +=  flow * theta'
        end        
    end
    
    weights
end

# 5.795 ms (72350 allocations: 6.58 MiB)
@inline function do_nothing_test(lc::LogisticCircuit,  classes::Int, data)
    likelihoods = zeros(num_examples(data), classes)
    foreach(or_nodes(lc)) do ln
        foreach(eachrow(ln.thetas), children(ln)) do theta, c
            nothing
        end        
    end
    @avx @. likelihoods = 1.0 / (1.0 + exp(-likelihoods))
    likelihoods
end

# 1.574 s (193840 allocations: 6.98 GiB)
@inline function access_flow_test(lc::LogisticCircuit,  classes::Int, data)
    foreach(or_nodes(lc)) do ln
        foreach(eachrow(ln.thetas), children(ln)) do theta, c
            flow = Float32.(downflow(ln, c)) 
        end        
    end
    nothing
end

# 2.943 s (82272 allocations: 6.74 MiB)
@inline function dummy_calculation_test(lc::LogisticCircuit,  classes::Int, data)
    likelihoods = zeros(num_examples(data), classes)
    foreach(or_nodes(lc)) do ln
        foreach(eachrow(ln.thetas), children(ln)) do theta, c
            @avx @. likelihoods += likelihoods
        end        
    end
    @avx @. likelihoods = 1.0 / (1.0 + exp(-likelihoods))
    likelihoods
end

# 4.790 s (193843 allocations: 6.99 GiB)
@inline function dummy_calculation_test2(lc::LogisticCircuit,  classes::Int, data)
    likelihoods = zeros(num_examples(data), classes)
    foreach(or_nodes(lc)) do ln
        foreach(eachrow(ln.thetas), children(ln)) do theta, c
            flow = Float32.(downflow(ln, c))
            @avx @. likelihoods += likelihoods
        end        
    end
    @avx @. likelihoods = 1.0 / (1.0 + exp(-likelihoods))
    likelihoods
end

@inline downflow(or_parent::Logistic⋁Node, c) = 
    (c.data isa UpDownFlow1) ? c.data.downflow : flow_and(or_parent.data.downflow, c.data, or_parent.data.upflow)

@inline flow_and(downflow_n::BitVector, c_flow::UpDownFlow2, upflow_n::BitVector) = 
    @. downflow_n & c_flow.prime_flow & c_flow.sub_flow

@inline flow_and(downflow_n::Vector{<:AbstractFloat}, c_flow::UpDownFlow2, upflow_n::Vector{<:AbstractFloat}) = 
    @avx @. downflow_n * c_flow.prime_flow * make_finite(c_flow.sub_flow/upflow_n)

@inline make_finite(x::T) where T = vifelse(isfinite(x), x, zero(T))


"""
Class Predictions
"""
@inline function predict_class(lc::LogisticCircuit, classes::Int, data)
    class_likelihoods = class_likelihood_per_instance(lc, classes, data)
    predict_class(class_likelihoods)
end

@inline function predict_class(class_likelihoods::AbstractMatrix)
    _, mxindex = findmax(class_likelihoods; dims=2)
    dropdims(getindex.(mxindex, 2); dims=2)
end

"""
Prediction accuracy
"""
@inline accuracy(predicted_class::Vector, labels) = 
    Float64(sum(@. predicted_class == labels)) / length(labels)

@inline accuracy(lc::LogisticCircuit, classes::Int, data, labels) = 
    accuracy(predict_class(lc, classes, data), labels)

@inline accuracy(class_likelihoods::AbstractMatrix, labels) = 
    accuracy(predict_class(class_likelihoods), labels)

