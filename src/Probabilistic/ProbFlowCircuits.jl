#####################

#TODO This code seems to assume logspace flows as floating point numbers. if so, enforca that on type F
function marginal_pass_up(circuit::UpFlowΔ{O,F}, data::XData{E}) where {E <: eltype(F)} where {O,F}
    resize_flows(circuit, num_examples(data))
    for n in circuit
        marginal_pass_up_node(n, data)
    end
end

marginal_pass_up_node(n::UpFlowΔNode, ::PlainXData) = ()

function marginal_pass_up_node(n::UpFlowLiteral{O,F}, data::PlainXData{E}) where {E <: eltype(F)} where {O,F}
    pass_up_node(n, data)
    # now override missing values by 1
    npr = pr(n)
    missing_features = feature_matrix(data)[:,variable(n)] .< zero(eltype(F))
    npr[missing_features] .= 1
    npr .= log.( npr .+ 1e-300 )
end

function marginal_pass_up_node(n::UpFlow⋀Cached, ::PlainXData)
    pr(n) .= pr(n.children[1])
    for c in n.children[2:end]
        pr(n) .+= pr(c)
    end
end

function marginal_pass_up_node(n::UpFlow⋁Cached, ::PlainXData)
    # A simple for loop seems to be way faster than logsumexp because of memory allocations are much lower.
    pr(n) .= 0.0
    for i=1:length(n.children)
        pr(n) .+= exp.( pr(n.children[i]) .+ (n.origin.log_thetas[i])  )
    end
    pr(n) .= log.(pr(n))

    ## logsumexp version
    # npr = pr(n)
    # log_thetai_pi = [ pr(n.children[i]) .+ (n.origin.log_thetas[i]) for i=1:length(n.children)]
    # ll = sum.(map((lls...) -> logsumexp([lls...]), log_thetai_pi...)) 
    # npr .= ll
end


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

marginal_pass_down_node(n::DownFlowΔNode) = () # do nothing
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
                sink.downflow .= downflow(n) .* exp.(grand_origin(n).log_thetas[ind] .+ pr(origin(c)) .- pr(origin(n)) )
                sink.in_progress = true
            else
                sink.downflow .+= downflow(n) .* exp.(grand_origin(n).log_thetas[ind] .+ pr(origin(c)) .- pr(origin(n))) 
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