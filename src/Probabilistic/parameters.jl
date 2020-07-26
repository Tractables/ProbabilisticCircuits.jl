export estimate_parameters

"""
Maximum likilihood estimation of parameters given data
"""
function estimate_parameters(pc::ProbCircuit, data; pseudocount::Float64)
    @assert isbinarydata(data)
    compute_flows(pc, data)
    foreach(pc) do pn
        if pn isa Prob⋁Node
            if num_children(pn) == 1
                pn.log_thetas .= 0.0
            else
                smoothed_flow = Float64(sum(get_downflow(pn))) + pseudocount
                uniform_pseudocount = pseudocount / num_children(pn)
                children_flows = map(c -> sum(get_downflow(pn, c)), children(pn))
                @. pn.log_thetas = log((children_flows + uniform_pseudocount) / smoothed_flow)
                @assert isapprox(sum(exp.(pn.log_thetas)), 1.0, atol=1e-6) "Parameters do not sum to one locally"
                # normalize away any leftover error
                pn.log_thetas .-= logsumexp(pn.log_thetas)
            end
        end
    end
end


# TODO add em paramaters learning 