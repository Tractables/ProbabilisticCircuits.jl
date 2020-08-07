export learn_parameters

using LogicCircuits: compute_flows
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
    error = class_probs .- one_hot_labels

    foreach(lc) do ln
        if ln isa Logistic⋁Node
            #TODO; check whether einsum would speed up calculations here
            # For each class. orig.thetas is 2D so used eachcol
            for (class, thetaC) in enumerate(eachcol(ln.thetas))
                for (idx, c) in enumerate(children(ln))
                    down_flow = Float64.(downflow(ln, c))
                    thetaC[idx] -= step_size * sum(error[:, class] .* down_flow) / length(down_flow)
                end
            end
        end
    end
    
    nothing
end


