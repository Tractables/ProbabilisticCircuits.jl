export learn_parameters

using LogicCircuits: compute_flows, or_nodes
using LoopVectorization: @avx

"""
Maximum likilihood estimation of parameters given data through gradient descent
"""
function learn_parameters(lc::LogisticCircuit, classes::Int, data, labels; num_epochs=30, step_size=0.1, flows_computed=false)

    @inline function one_hot(labels::Vector, classes::Int)        
        one_hot_labels = zeros(length(labels), classes)
        for (i, j) in enumerate(labels)
            one_hot_labels[i, j] = 1.0
        end
        one_hot_labels
    end

    one_hot_labels = one_hot(labels, classes)
    if !flows_computed
        compute_flows(lc, data)
    end

    for _ = 1:num_epochs
        class_probs = class_conditional_likelihood_per_instance(lc, classes, data; flows_computed=true)
        update_parameters(lc, class_probs, one_hot_labels)
    end

    nothing
end


@inline function update_parameters(lc::LogisticCircuit, class_probs, one_hot_labels; step_size=0.1)
    num_samples = Float64(size(one_hot_labels)[1])
    error = class_probs .- one_hot_labels
    
    foreach(or_nodes(lc)) do ln
        foreach(eachrow(ln.thetas), children(ln)) do theta, c
            flow = Float64.(downflow(ln, c))
            @avx update_amount = flow' * error / num_samples * step_size
            update_amount = dropdims(update_amount; dims=1)
            @avx @. theta -= update_amount
        end
    end
    
    nothing
end


