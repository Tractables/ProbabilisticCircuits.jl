
#####################

#TODO This code seems to assume logspace flows as floating point numbers. if so, enforca that on type F
function marginal_pass_up(circuit::FlowCircuit△{F}, data::XData{E}) where {E <: eltype(F)} where F
    resize_flows(circuit, num_examples(data))
    for n in circuit
        marginal_pass_up_node(n, data)
    end
end

marginal_pass_up_node(n::FlowCircuitNode, ::PlainXData) = ()

function marginal_pass_up_node(n::FlowLiteral{F}, data::PlainXData{E}) where {E <: eltype(F)} where F
    pass_up_node(n, data)
    # now override missing values by 1
    npr = pr(n)
    missing_features = feature_matrix(data)[:,variable(n)] .< zero(eltype(F))
    npr[missing_features] .= 1
    npr .= log.( npr .+ 1e-300 )
end

function marginal_pass_up_node(n::Flow⋀Cached, ::PlainXData)
    npr = pr(n)
    npr .= pr(n.children[1])
    for c in n.children[2:end]
        npr .+= pr(c)
    end
end

function marginal_pass_up_node(n::Flow⋁Cached, ::PlainXData)
    npr = pr(n)
    log_thetai_pi = [ pr(n.children[i]) .+ (n.origin.log_thetas[i]) for i=1:length(n.children)]
    ll = sum.(map((lls...) -> logsumexp([lls...]), log_thetai_pi...)) 
    npr .= ll
end


##### marginal_pass_down

function marginal_pass_down(circuit::FlowCircuit△{F}) where {F}
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

marginal_pass_down_node(n::FlowCircuitNode) = () # do nothing
marginal_pass_down_node(n::FlowLiteral) = ()

function marginal_pass_down_node(n::Flow⋀Cached)
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

function marginal_pass_down_node(n::Flow⋁Cached)
    # todo(pashak) might need some changes, not tested, also to convert to logexpsum later
    # downflow(n) = EF_n(e), the EF for edges or leaves are note stored
    for (ind, c) in enumerate(n.children)
        for sink in downflow_sinks(c)
            if !sink.in_progress
                sink.downflow .= downflow(n) .* exp.(n.origin.log_thetas[ind] .+ pr(c) .- pr(n) )
                sink.in_progress = true
            else
                sink.downflow .+= downflow(n) .* exp.(n.origin.log_thetas[ind] .+ pr(c) .- pr(n)) 
            end
        end
    end
end

#### marginal_pass_up_down

function marginal_pass_up_down(circuit::FlowCircuit△{F}, data::XData{E}) where {E <: eltype(F)} where F
    @assert !(E isa Bool)
    marginal_pass_up(circuit, data)
    marginal_pass_down(circuit)
end