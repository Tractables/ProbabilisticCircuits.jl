import LogicCircuits: evaluate
using StatsFuns: logsumexp

# evaluate a circuit as a function
function (root::ProbCircuit)(data)
    evaluate(root, data)
end

"Container for circuit flows represented as a float vector"
const UpFlow1 = Vector{Float64}


"Container for circuit flows represented as an implicit conjunction of a prime and sub float vector (saves memory allocations in circuits with many binary conjunctions)"
struct UpFlow2
    prime_flow::Vector{Float64}
    sub_flow::Vector{Float64}
end

const UpFlow = Union{UpFlow1,UpFlow2}

@inline UpFlow(elems::UpFlow1) = elems

@inline UpFlow(elems::UpFlow2) =
    elems.prime_flow .+ elems.sub_flow

import Base.length
length(elems::UpFlow2) = length(UpFlow(elems))

function evaluate(root::ProbCircuit, data;
                   nload = nload, nsave = nsave, reset=true)::Vector{Float64}
    n_ex::Int = num_examples(data)
    ϵ = 1e-300

    @inline f_lit(n) = begin
        uf = convert(Vector{Int8}, feature_values(data, variable(n)))
        if ispositive(origin(n))
            uf[uf.==-1] .= 1
        else
            uf .= 1 .- uf
            uf[uf.==2] .= 1
        end
        uf = convert(Vector{Float64}, uf)
        uf .= log.(uf .+ ϵ)
    end
    
    @inline f_con(n) = begin
        uf = istrue(origin(n)) ? ones(Float64, n_ex) : zeros(Float64, n_ex)
        uf .= log.(uf .+ ϵ)
    end
    
    @inline fa(n, call) = begin
        if num_children(n) == 1
            return UpFlow1(call(@inbounds children(n)[1]))
        else
            c1 = call(@inbounds children(n)[1])::UpFlow
            c2 = call(@inbounds children(n)[2])::UpFlow
            if num_children(n) == 2 && c1 isa UpFlow1 && c2 isa UpFlow1 
                UpFlow2(c1, c2) # no need to allocate a new BitVector
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
            return UpFlow1(call(@inbounds children(n)[1]))
        else
            log_thetas = n.log_thetas
            c1 = call(@inbounds children(n)[1])::UpFlow
            c2 = call(@inbounds children(n)[2])::UpFlow
            x = flowop(c1, log_thetas[1], c2, log_thetas[2], logsumexp)
            for (i, c) in enumerate(children(n)[3:end])
                accumulate(x, call(c), log_thetas[i+2], logsumexp)
            end
            return x
        end
    end
    
    # ensure flow us Flow1 at the root, even when it's a conjunction
    root_flow = UpFlow1(foldup(root, f_con, f_lit, fa, fo, UpFlow; nload, nsave, reset))
    return nsave(root, root_flow)
end

@inline flowop(x::UpFlow, y::UpFlow, op)::UpFlow1 =
    op.(UpFlow(x), UpFlow(y))

@inline flowop(x::UpFlow, w1::Float64, y::UpFlow, w2::Float64, op)::UpFlow1 =
    op.(UpFlow(x) .+ w1, UpFlow(y) .+ w2)

import Base.accumulate
@inline accumulate(x::UpFlow, v::UpFlow, op) = 
    @inbounds @. x = op($UpFlow(x), $UpFlow(v)); nothing

@inline accumulate(x::UpFlow, v::UpFlow, w::Float64, op) = 
    @inbounds @. x = op($UpFlow(x), $UpFlow(v) + w); nothing
