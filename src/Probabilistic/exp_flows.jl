export evaluate_exp, compute_exp_flows, get_downflow, get_upflow, get_exp_downflow, get_exp_upflow


using StatsFuns: logsumexp

# TODO move to LogicCircuits
# TODO downflow struct
using LogicCircuits: UpFlow, UpFlow1, UpDownFlow, UpDownFlow1, UpDownFlow2

"""
Get upflow from logic circuit
"""
@inline get_upflow(n::LogicCircuit) = get_upflow(n.data)
@inline get_upflow(elems::UpDownFlow1) = elems.upflow
@inline get_upflow(elems::UpFlow) = UpFlow1(elems)

"""
Get the node/edge flow from logic circuit
"""
function get_downflow(n::LogicCircuit; root=nothing)::BitVector
    downflow(x::UpDownFlow1) = x.downflow
    downflow(x::UpDownFlow2) = begin
        ors = or_nodes(root)
        p = findall(p -> n in children(p), ors)
        @assert length(p) == 1
        get_downflow(ors[p[1]], n)
    end
    downflow(n.data)
end

function get_downflow(n::LogicCircuit, c::LogicCircuit)::BitVector
    @assert !is⋁gate(c) && is⋁gate(n) && c in children(n)
    get_downflow(n) .& get_upflow(c)
end

#####################
# performance-critical queries related to circuit flows
#####################

"Container for circuit flows represented as a float vector"
const ExpUpFlow1 = Vector{Float64}

"Container for circuit flows represented as an implicit conjunction of a prime and sub float vector (saves memory allocations in circuits with many binary conjunctions)"
struct ExpUpFlow2
    prime_flow::Vector{Float64}
    sub_flow::Vector{Float64}
end

const ExpUpFlow = Union{ExpUpFlow1,ExpUpFlow2}

@inline ExpUpFlow1(elems::ExpUpFlow1) = elems
@inline ExpUpFlow1(elems::ExpUpFlow2) = elems.prime_flow .+ elems.sub_flow

function evaluate_exp(root::ProbCircuit, data;
                   nload = nload, nsave = nsave, reset=true)::Vector{Float64}
    n_ex::Int = num_examples(data)
    ϵ = 1e-300

    @inline f_lit(n) = begin
        uf = convert(Vector{Int8}, feature_values(data, variable(n)))
        if ispositive(n)
            uf[uf.==-1] .= 1
        else
            uf .= 1 .- uf
            uf[uf.==2] .= 1
        end
        uf = convert(Vector{Float64}, uf)
        uf .= log.(uf .+ ϵ)
    end
    
    @inline f_con(n) = begin
        uf = istrue(n) ? ones(Float64, n_ex) : zeros(Float64, n_ex)
        uf .= log.(uf .+ ϵ)
    end
    
    @inline fa(n, call) = begin
        if num_children(n) == 1
            return ExpUpFlow1(call(@inbounds children(n)[1]))
        else
            c1 = call(@inbounds children(n)[1])::ExpUpFlow
            c2 = call(@inbounds children(n)[2])::ExpUpFlow
            if num_children(n) == 2 && c1 isa ExpUpFlow1 && c2 isa ExpUpFlow1 
                return ExpUpFlow2(c1, c2) # no need to allocate a new BitVector
            end
            x = flowop(c1, c2, +)
            for c in children(n)[3:end]
                accumulate(x, call(c), +)
            end
            return x
        end
    end
    
    @inline fo(n, call) = begin
        if num_children(n) == 1
            return ExpUpFlow1(call(@inbounds children(n)[1]))
        else
            log_thetas = n.log_thetas
            c1 = call(@inbounds children(n)[1])::ExpUpFlow
            c2 = call(@inbounds children(n)[2])::ExpUpFlow
            x = flowop(c1, log_thetas[1], c2, log_thetas[2], logsumexp)
            for (i, c) in enumerate(children(n)[3:end])
                accumulate(x, call(c), log_thetas[i+2], logsumexp)
            end
            return x
        end
    end
    
    # ensure flow us Flow1 at the root, even when it's a conjunction
    root_flow = ExpUpFlow1(foldup(root, f_con, f_lit, fa, fo, ExpUpFlow; nload, nsave, reset))
    return nsave(root, root_flow)
end

@inline flowop(x::ExpUpFlow, y::ExpUpFlow, op)::ExpUpFlow1 =
    op.(ExpUpFlow1(x), ExpUpFlow1(y))

@inline flowop(x::ExpUpFlow, w1::Float64, y::ExpUpFlow, w2::Float64, op)::ExpUpFlow1 =
    op.(ExpUpFlow1(x) .+ w1, ExpUpFlow1(y) .+ w2)

import Base.accumulate
@inline accumulate(x::ExpUpFlow1, v::ExpUpFlow, op) = 
    @inbounds @. x = op($ExpUpFlow1(x), $ExpUpFlow1(v)); nothing

@inline accumulate(x::ExpUpFlow1, v::ExpUpFlow, w::Float64, op) = 
    @inbounds @. x = op($ExpUpFlow1(x), $ExpUpFlow1(v) + w); nothing


#####################
# downward pass
#####################

struct ExpUpDownFlow1
    upflow::ExpUpFlow1
    downflow::Vector{Float64}
    ExpUpDownFlow1(upf::ExpUpFlow1) = new(upf, log.(zeros(Float64, length(upf)) .+ 1e-300))
end

const ExpUpDownFlow2 = ExpUpFlow2

const ExpUpDownFlow = Union{ExpUpDownFlow1, ExpUpDownFlow2}


function compute_exp_flows(circuit::ProbCircuit, data)

    # upward pass
    @inline upflow!(n, v) = begin
        n.data = (v isa ExpUpFlow1) ? ExpUpDownFlow1(v) : v
        v
    end

    @inline upflow(n) = begin
        d = n.data::ExpUpDownFlow
        (d isa ExpUpDownFlow1) ? d.upflow : d
    end

    evaluate_exp(circuit, data; nload=upflow, nsave=upflow!, reset=false)
    
    # downward pass

    @inline downflow(n) = (n.data::ExpUpDownFlow1).downflow
    @inline isfactorized(n) = n.data::ExpUpDownFlow isa ExpUpDownFlow2

    downflow(circuit) .= 0.0

    foreach_down(circuit; setcounter=false) do n
        if isinner(n) && !isfactorized(n)
            downflow_n = downflow(n)
            upflow_n = upflow(n)
            for ite in 1 : num_children(n)
                c = children(n)[ite]
                log_theta = is⋀gate(n) ? 0.0 : n.log_thetas[ite]
                if isfactorized(c)
                    upflow2_c = c.data::ExpUpDownFlow2
                    # propagate one level further down
                    for i = 1:2
                        downflow_c = downflow(@inbounds children(c)[i])
                        accumulate(downflow_c, downflow_n .+ log_theta .+ upflow2_c.prime_flow 
                        .+ upflow2_c.sub_flow .- upflow_n, logsumexp)
                    end
                else
                    upflow1_c = (c.data::ExpUpDownFlow1).upflow
                    downflow_c = downflow(c)
                    accumulate(downflow_c, downflow_n .+ log_theta .+ upflow1_c .- upflow_n, logsumexp)
                end
            end 
        end
        nothing
    end
    nothing
end


"""
Get upflow of a probabilistic circuit
"""
@inline get_exp_upflow(pc::ProbCircuit) = get_exp_upflow(pc.data)
@inline get_exp_upflow(elems::ExpUpDownFlow1) = elems.upflow
@inline get_exp_upflow(elems::ExpUpFlow) = ExpUpFlow1(elems)

"""
Get the node/edge downflow from probabilistic circuit
"""
function get_exp_downflow(n::ProbCircuit; root=nothing)::Vector{Float64}
    downflow(x::ExpUpDownFlow1) = x.downflow
    downflow(x::ExpUpDownFlow2) = begin
        ors = or_nodes(root)
        p = findall(p -> n in children(p), ors)
        @assert length(p) == 1
        get_exp_downflow(ors[p[1]], n)
    end
    downflow(n.data)
end

function get_exp_downflow(n::ProbCircuit, c::ProbCircuit)::Vector{Float64}
    @assert !is⋁gate(c) && is⋁gate(n) && c in children(n)
    log_theta = n.log_thetas[findfirst(x -> x == c, children(n))]
    return get_exp_downflow(n) .+ log_theta .+ get_exp_upflow(c) .- get_exp_upflow(n)
end