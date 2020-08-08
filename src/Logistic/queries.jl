export class_conditional_weights_per_instance, 
       class_conditional_likelihood_per_instance, 
       downflow, accuracy, predict_class

using LogicCircuits: UpDownFlow1, flow_and, or_nodes
using ..Probabilistic: get_downflow, get_upflow
using LoopVectorization: @avx

"""
Class Conditional Weights
"""
# This is the old implementation. It is retained to pass the exptation tests.
function class_conditional_weights_per_instance(lc::LogisticCircuit,  classes::Int, data; flows_computed=false)
    if !flows_computed
        compute_flows(lc, data)
    end

    likelihoods = zeros(num_examples(data), classes)
    foreach(or_nodes(lc)) do ln
        # For each class. orig.thetas is 2D so used eachcol
        for (class, thetaC) in enumerate(eachcol(ln.thetas))
            foreach(children(ln), thetaC) do c, theta
                likelihoods[:, class] .+= Float64.(get_downflow(ln) .& get_upflow(c)) .* theta
            end
        end
    end
    likelihoods
end

"""
Class Conditional Probability
"""
@inline function class_conditional_likelihood_per_instance(lc::LogisticCircuit,  classes::Int, data; flows_computed=false)
    if !flows_computed
        compute_flows(lc, data)
    end
    
    likelihoods = zeros(num_examples(data), classes)
    foreach(or_nodes(lc)) do ln
        foreach(eachrow(ln.thetas), children(ln)) do theta, c
            flow = Float64.(downflow(ln, c))
            @avx @. likelihoods +=  flow * theta'
        end        
    end
    
    @avx @. likelihoods = 1.0 / (1.0 + exp(-likelihoods))
    likelihoods
end

@inline downflow(or_parent::Logistic⋁Node, c) = 
    (c.data isa UpDownFlow1) ? c.data.downflow : flow_and(or_parent.data.downflow, c.data)

"""
Class Predictions
"""
@inline function predict_class(lc::LogisticCircuit, classes::Int, data)
    class_likelihoods = class_conditional_likelihood_per_instance(lc, classes, data)
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

