export class_conditional_likelihood_per_instance

using ..Probabilistic: get_downflow, get_upflow
"""
Class Conditional Probability
"""
function class_conditional_likelihood_per_instance(lc::LogisticCircuit,  classes::Int, data)
    compute_flows(lc.origin, data)
    likelihoods = zeros(num_examples(data), classes)
    foreach(lc) do ln
        if ln isa Logistic⋁Node
            # For each class. orig.thetas is 2D so used eachcol
            for (idx, thetaC) in enumerate(eachcol(ln.thetas))
                foreach(ln.children, thetaC) do c, theta
                    likelihoods[:, idx] .+= Float64.(get_downflow(ln.origin) .& get_upflow(c.origin)) .* theta
                end
            end
        end
    end
    likelihoods
end

