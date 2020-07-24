##### marginal_pass_down

function marginal_pass_down(circuit::DownFlowΔ{O,F}) where {O,F}
    resize_flows(circuit, flow_length(origin(circuit)))
    for n in circuit
        reset_downflow_in_progress(n)
    end
    for downflow in downflow_sinks(circuit[end])
        # initialize root flows to 1
        downflow.downflow .= one(eltype(F))
    end
    for n in Iterators.reverse(circuit)
        marginal_pass_down_node(n)
    end
end

marginal_pass_down_node(n::DownFlowNode) = () # do nothing
marginal_pass_down_node(n::DownFlowLeaf) = ()

function marginal_pass_down_node(n::DownFlow⋀Cached)
    # todo(pashak) might need some changes, not tested, also to convert to logexpsum later
     # downflow(n) = EF_n(e), the EF for edges or leaves are note stored
    for c in n.children
        for sink in downflow_sinks(c)
            if !sink.in_progress
                sink.downflow .= downflow(n)
                sink.in_progress = true
            else
                sink.downflow .+= downflow(n)
            end
        end
    end
end

function marginal_pass_down_node(n::DownFlow⋁Cached)
    # todo(pashak) might need some changes, not tested, also to convert to logexpsum later
    # downflow(n) = EF_n(e), the EF for edges or leaves are note stored
    for (ind, c) in enumerate(n.children)
        for sink in downflow_sinks(c)
            if !sink.in_progress
                sink.downflow .= downflow(n) .* exp.(prob_origin(n).log_thetas[ind] .+ pr(origin(c)) .- pr(origin(n)) )
                sink.in_progress = true
            else
                sink.downflow .+= downflow(n) .* exp.(prob_origin(n).log_thetas[ind] .+ pr(origin(c)) .- pr(origin(n)))
            end
        end
    end
end

#### marginal_pass_up_down

function marginal_pass_up_down(circuit::DownFlowΔ{O,F}, data::XData{E}) where {E <: eltype(F)} where {O,F}
    @assert !(E isa Bool)
    marginal_pass_up(origin(circuit), data)
    marginal_pass_down(circuit)
end
