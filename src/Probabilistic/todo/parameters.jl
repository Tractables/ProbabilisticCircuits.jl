function normalize_parameters(pc::ProbCircuit)
    for or in or_nodes(pc)
        or.log_thetas .= 1 ./ length(or.log_thetas)
    end
end


function estimate_parameters2(pc::ProbΔ, data::XData{Bool}; pseudocount::Float64)
    Logic.pass_up_down2(pc, data)
    w = (data isa PlainXData) ? nothing : weights(data)
    estimate_parameters_cached2(pc, w; pseudocount=pseudocount)
end

function estimate_parameters_cached2(pc::ProbΔ, w; pseudocount::Float64)
    flow(n) = Float64(sum(sum(n.data)))
    children_flows(n) = sum.(map(c -> c.data[1] .& n.data[1], children(n)))

    if issomething(w)
        flow_w(n) = sum(Float64.(n.data[1]) .* w)
        children_flows_w(n) = sum.(map(c -> Float64.(c.data[1] .& n.data[1]) .* w, children(n)))
        flow = flow_w
        children_flows = children_flows_w
    end

    estimate_parameters_node2(n::ProbNode) = ()
    function estimate_parameters_node2(n::Prob⋁)
        if num_children(n) == 1
            n.log_thetas .= 0.0
        else
            smoothed_flow = flow(n) + pseudocount
            uniform_pseudocount = pseudocount / num_children(n)
            n.log_thetas .= log.((children_flows(n) .+ uniform_pseudocount) ./ smoothed_flow)
            @assert isapprox(sum(exp.(n.log_thetas)), 1.0, atol=1e-6) "Parameters do not sum to one locally"
            # normalize away any leftover error
            n.log_thetas .- logsumexp(n.log_thetas)
        end
    end

    foreach(estimate_parameters_node2, pc)
end




function estimate_parameters(pc::ProbΔ, data::XBatches{Bool}; pseudocount::Float64)
    estimate_parameters(AggregateFlowΔ(pc, aggr_weight_type(data)), data; pseudocount=pseudocount)
end

function estimate_parameters(afc::AggregateFlowΔ, data::XBatches{Bool}; pseudocount::Float64)
    @assert feature_type(data) == Bool "Can only learn probabilistic circuits on Bool data"
    @assert (afc[end].origin isa ProbNode) "AggregateFlowΔ must originate in a ProbΔ"
    collect_aggr_flows(afc, data)
    estimate_parameters_cached(afc; pseudocount=pseudocount)
    afc
end

function estimate_parameters(fc::FlowΔ, data::XBatches{Bool}; pseudocount::Float64)
    @assert feature_type(data) == Bool "Can only learn probabilistic circuits on Bool data"
    @assert (prob_origin(afc[end]) isa ProbNode) "FlowΔ must originate in a ProbΔ"
    collect_aggr_flows(fc, data)
    estimate_parameters_cached(origin(fc); pseudocount=pseudocount)
end

 # turns aggregate statistics into theta parameters
function estimate_parameters_cached(afc::AggregateFlowΔ; pseudocount::Float64)
    foreach(n -> estimate_parameters_node(n; pseudocount=pseudocount), afc)
end

estimate_parameters_node(::AggregateFlowNode; pseudocount::Float64) = () # do nothing
function estimate_parameters_node(n::AggregateFlow⋁; pseudocount)
    origin = n.origin::Prob⋁
    if num_children(n) == 1
        origin.log_thetas .= 0.0
    else
        smoothed_aggr_flow = (n.aggr_flow + pseudocount)
        uniform_pseudocount = pseudocount / num_children(n)
        origin.log_thetas .= log.( (n.aggr_flow_children .+ uniform_pseudocount) ./ smoothed_aggr_flow )
        @assert isapprox(sum(exp.(origin.log_thetas)), 1.0, atol=1e-6) "Parameters do not sum to one locally: $(exp.(origin.log_thetas)), estimated from $(n.aggr_flow) and $(n.aggr_flow_children). Did you actually compute the aggregate flows?"
        #normalize away any leftover error
        origin.log_thetas .- logsumexp(origin.log_thetas)
    end
end

##################
# Sampling from a psdd
##################

"""
Sample from a PSDD without any evidence
"""
function sample(circuit::ProbΔ)::AbstractVector{Bool}
    inst = Dict{Var,Int64}()
    simulate(circuit[end], inst)
    len = length(keys(inst))
    ans = Vector{Bool}()
    for i = 1:len
        push!(ans, inst[i])
    end
    ans
end

# Uniformly sample based on the probability of the items
# and return the selected index
function sample(probs::AbstractVector{<:Number})::Int32
    z = sum(probs)
    q = rand() * z
    cur = 0.0
    for i = 1:length(probs)
        cur += probs[i]
        if q <= cur
            return i
        end
    end
    return length(probs)
end

function simulate(node::ProbLiteral, inst::Dict{Var,Int64})
    if ispositive(node)
        inst[variable(node.origin)] = 1
    else
        inst[variable(node.origin)] = 0
    end
end

function simulate(node::Prob⋁, inst::Dict{Var,Int64})
    idx = sample(exp.(node.log_thetas))
    simulate(node.children[idx], inst)
end
function simulate(node::Prob⋀, inst::Dict{Var,Int64})
    for child in node.children
        simulate(child, inst)
    end
end

"""
Sampling with Evidence from a psdd.
Internally would call marginal pass up on a newly generated flow circuit.
"""
function sample(circuit::ProbΔ, evidence::PlainXData{Int8})::AbstractVector{Bool}
    opts= (compact⋀=false, compact⋁=false)
    flow_circuit = UpFlowΔ(circuit, 1, Float64, opts)
    marginal_pass_up(flow_circuit, evidence)
    sample(flow_circuit)
end

"""
Sampling with Evidence from a psdd.
Assuming already marginal pass up has been done on the flow circuit.
"""
function sample(circuit::UpFlowΔ)::AbstractVector{Bool}
    inst = Dict{Var,Int64}()
    simulate2(circuit[end], inst)
    len = length(keys(inst))
    ans = Vector{Bool}()
    for i = 1:len
        push!(ans, inst[i])
    end
    ans
end

function simulate2(node::UpFlowLiteral, inst::Dict{Var,Int64})
    if ispositive(node)
        #TODO I don't think we need these 'grand_origin' parts below
        inst[variable(grand_origin(node))] = 1
    else
        inst[variable(grand_origin(node))] = 0
    end
end

function simulate2(node::UpFlow⋁, inst::Dict{Var,Int64})
    prs = [ pr(ch)[1] for ch in children(node) ]
    idx = sample(exp.(node.origin.log_thetas .+ prs))
    simulate2(children(node)[idx], inst)
end

function simulate2(node::UpFlow⋀, inst::Dict{Var,Int64})
    for child in children(node)
        simulate2(child, inst)
    end
end