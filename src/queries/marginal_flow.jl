using StatsFuns: logsumexp, log1pexp

using CUDA: CUDA, @cuda
using DataFrames: DataFrame
using LoopVectorization: @avx
using LogicCircuits: balance_threads

export marginal, marginal_all

#####################
# Circuit marginal evaluation
#####################
  
# evaluate a probabilistic circuit as a function
function (root::ProbCircuit)(data...)
    marginal(root, data...)
end

"Evaluate marginals of the circuit bottom-up for a given input"
marginal(root::ProbCircuit, data::Union{Real,Missing}...) =
    marginal(root, collect(Union{Bool,Missing}, data))

marginal(root::ProbCircuit, data::Union{Vector{Union{Bool,Missing}},CuVector{UInt8}}) =
    marginal(root, DataFrame(reshape(data, 1, :)))[1]

marginal(circuit::ProbCircuit, data::DataFrame) =
    marginal(same_device(ParamBitCircuit(circuit, data), data) , data)

function marginal(circuit::ParamBitCircuit, data::DataFrame)::AbstractVector
    marginal_all(circuit,data)[:,end]
end

#####################
# Circuit evaluation of *all* nodes in circuit
#####################

"Evaluate the probabilistic circuit bottom-up for a given input and return the marginal probability value of all nodes"
function marginal_all(circuit::ParamBitCircuit, data, reuse=nothing)
    @assert num_features(data) == num_features(circuit) 
    @assert isbinarydata(data)
    values = init_marginal(data, reuse, num_nodes(circuit))
    marginal_layers(circuit, values)
    return values
end

"Initialize values from the data (data frames)"
function init_marginal(data, reuse, num_nodes)
    flowtype = isgpu(data) ? CuMatrix{Float64} : Matrix{Float64}
    values = similar!(reuse, flowtype, num_examples(data), num_nodes)
    @views values[:,LogicCircuits.TRUE_BITS] .= log(one(Float64))
    @views values[:,LogicCircuits.FALSE_BITS] .= log(zero(Float64))
    # here we should use a custom CUDA kernel to extract Float marginals from bit vectors
    # for now the lazy solution is to move everything to the CPU and do the work there...
    data_cpu = to_cpu(data)
    for i=1:num_features(data)
        marg_pos::Vector{Float64} = log.(coalesce.(data_cpu[:,i], 1.0))
        marg_neg::Vector{Float64} = log.(coalesce.(1.0 .- data_cpu[:,i], 1.0))
        values[:,2+i] .= same_device(marg_pos, values)
        values[:,2+num_features(data)+i] .= same_device(marg_neg, values)
    end
    return values
end

# upward pass helpers on CPU

"Compute marginals on the CPU (SIMD & multi-threaded)"
function marginal_layers(circuit::ParamBitCircuit, values::Matrix)
    bc = circuit.bitcircuit
    els = bc.elements
    pars = circuit.params
    for layer in bc.layers
        Threads.@threads for dec_id in layer
            j = @inbounds bc.nodes[1,dec_id]
            els_end = @inbounds bc.nodes[2,dec_id]
            if j == els_end
                assign_marginal(values, dec_id, els[2,j], els[3,j], pars[j])
                j += 1
            else
                assign_marginal(values, dec_id, els[2,j], els[3,j], els[2,j+1], els[3,j+1], pars[j], pars[j+1])
                j += 2
            end
            while j <= els_end
                accum_marginal(values, dec_id, els[2,j], els[3,j], pars[j])
                j += 1
            end
        end
    end
end

assign_marginal(v::Matrix{<:AbstractFloat}, i, e1p, e1s, p1) =
    @views @. @avx v[:,i] = v[:,e1p] + v[:,e1s] + p1

accum_marginal(v::Matrix{<:AbstractFloat}, i, e1p, e1s, p1) = begin
    @avx for j=1:size(v,1)
        @inbounds x = v[j,i]
        @inbounds y = v[j,e1p] + v[j,e1s] + p1
        Δ = ifelse(x == y, zero(Float64), abs(x - y))
        @inbounds  v[j,i] = max(x, y) + log1p(exp(-Δ))
    end
end

assign_marginal(v::Matrix{<:AbstractFloat}, i, e1p, e1s, e2p, e2s, p1, p2) = begin
    @avx for j=1:size(v,1)
        @inbounds x = v[j,e1p] + v[j,e1s] + p1
        @inbounds y = v[j,e2p] + v[j,e2s] + p2
        Δ = ifelse(x == y, zero(Float64), abs(x - y))
        @inbounds  v[j,i] = max(x, y) + log1p(exp(-Δ))
    end
end

# upward pass helpers on GPU

"Compute marginals on the GPU"
function marginal_layers(circuit::ParamBitCircuit, values::CuMatrix;  dec_per_thread = 8, log2_threads_per_block = 8)
    bc = circuit.bitcircuit
    CUDA.@sync for layer in bc.layers
        num_examples = size(values, 1)
        num_decision_sets = length(layer)/dec_per_thread
        num_threads =  balance_threads(num_examples, num_decision_sets, log2_threads_per_block)
        num_blocks = (ceil(Int, num_examples/num_threads[1]), 
                      ceil(Int, num_decision_sets/num_threads[2]))
        @cuda threads=num_threads blocks=num_blocks marginal_layers_cuda(layer, bc.nodes, bc.elements, circuit.params, values)
    end
end

"CUDA kernel for circuit evaluation"
function marginal_layers_cuda(layer, nodes, elements, params, values)
    index_x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    index_y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride_x = blockDim().x * gridDim().x
    stride_y = blockDim().y * gridDim().y
    for j = index_x:stride_x:size(values,1)
        for i = index_y:stride_y:length(layer)
            decision_id = @inbounds layer[i]
            k = @inbounds nodes[1,decision_id]
            els_end = @inbounds nodes[2,decision_id]
            @inbounds x = values[j, elements[2,k]] + values[j, elements[3,k]] + params[k]
            while k < els_end
                k += 1
                @inbounds y = values[j, elements[2,k]] + values[j, elements[3,k]] + params[k]
                Δ = ifelse(x == y, zero(Float64), CUDA.abs(x - y))
                x = max(x, y) + CUDA.log1p(CUDA.exp(-Δ))
            end
            values[j, decision_id] = x
        end
    end
    return nothing
end


# export evaluate_exp, compute_exp_flows, get_downflow, get_upflow, get_exp_downflow, get_exp_upflow



# # TODO move to LogicCircuits
# # TODO downflow struct
# using LogicCircuits: materialize, UpFlow, UpDownFlow, UpDownFlow1, UpDownFlow2

# """
# Get upflow from logic circuit
# """
# @inline get_upflow(n::LogicCircuit) = get_upflow(n.data)
# @inline get_upflow(elems::UpDownFlow1) = elems.upflow
# @inline get_upflow(elems::UpFlow) = materialize(elems)

# """
# Get the node/edge flow from logic circuit
# """
# function get_downflow(n::LogicCircuit; root=nothing)::BitVector
#     downflow(x::UpDownFlow1) = x.downflow
#     downflow(x::UpDownFlow2) = begin
#         ors = or_nodes(root)
#         p = findall(p -> n in children(p), ors)
#         @assert length(p) == 1
#         get_downflow(ors[p[1]], n)
#     end
#     downflow(n.data)
# end

# function get_downflow(n::LogicCircuit, c::LogicCircuit)::BitVector
#     @assert !is⋁gate(c) && is⋁gate(n) && c in children(n)
#     get_downflow(n) .& get_upflow(c)
# end

# #####################
# # performance-critical queries related to circuit flows
# #####################

# "Container for circuit flows represented as a float vector"
# const ExpUpFlow1 = Vector{Float64}

# "Container for circuit flows represented as an implicit conjunction of a prime and sub float vector (saves memory allocations in circuits with many binary conjunctions)"
# struct ExpUpFlow2
#     prime_flow::Vector{Float64}
#     sub_flow::Vector{Float64}
# end

# const ExpUpFlow = Union{ExpUpFlow1,ExpUpFlow2}

# @inline ExpUpFlow1(elems::ExpUpFlow1) = elems
# @inline ExpUpFlow1(elems::ExpUpFlow2) = elems.prime_flow .+ elems.sub_flow

# function evaluate_exp(root::ProbCircuit, data;
#                    nload = nload, nsave = nsave, reset=true)::Vector{Float64}
#     n_ex::Int = num_examples(data)
#     ϵ = 0.0

#     @inline f_lit(n) = begin
#         uf = convert(Vector{Int8}, feature_values(data, variable(n)))
#         if ispositive(n)
#             uf[uf.==-1] .= 1
#         else
#             uf .= 1 .- uf
#             uf[uf.==2] .= 1
#         end
#         uf = convert(Vector{Float64}, uf)
#         uf .= log.(uf .+ ϵ)
#     end
    
#     @inline f_con(n) = begin
#         uf = istrue(n) ? ones(Float64, n_ex) : zeros(Float64, n_ex)
#         uf .= log.(uf .+ ϵ)
#     end
    
#     @inline fa(n, call) = begin
#         if num_children(n) == 1
#             return ExpUpFlow1(call(@inbounds children(n)[1]))
#         else
#             c1 = call(@inbounds children(n)[1])::ExpUpFlow
#             c2 = call(@inbounds children(n)[2])::ExpUpFlow
#             if num_children(n) == 2 && c1 isa ExpUpFlow1 && c2 isa ExpUpFlow1 
#                 return ExpUpFlow2(c1, c2) # no need to allocate a new BitVector
#             end
#             x = flowop(c1, c2, +)
#             for c in children(n)[3:end]
#                 accumulate(x, call(c), +)
#             end
#             return x
#         end
#     end
    
#     @inline fo(n, call) = begin
#         if num_children(n) == 1
#             return ExpUpFlow1(call(@inbounds children(n)[1]))
#         else
#             log_probs = n.log_probs
#             c1 = call(@inbounds children(n)[1])::ExpUpFlow
#             c2 = call(@inbounds children(n)[2])::ExpUpFlow
#             x = flowop(c1, log_probs[1], c2, log_probs[2], logsumexp)
#             for (i, c) in enumerate(children(n)[3:end])
#                 accumulate(x, call(c), log_probs[i+2], logsumexp)
#             end
#             return x
#         end
#     end
    
#     # ensure flow us Flow1 at the root, even when it's a conjunction
#     root_flow = ExpUpFlow1(foldup(root, f_con, f_lit, fa, fo, ExpUpFlow; nload, nsave, reset))
#     return nsave(root, root_flow)
# end

# @inline flowop(x::ExpUpFlow, y::ExpUpFlow, op)::ExpUpFlow1 =
#     op.(ExpUpFlow1(x), ExpUpFlow1(y))

# @inline flowop(x::ExpUpFlow, w1::Float64, y::ExpUpFlow, w2::Float64, op)::ExpUpFlow1 =
#     op.(ExpUpFlow1(x) .+ w1, ExpUpFlow1(y) .+ w2)

# import Base.accumulate
# @inline accumulate(x::ExpUpFlow1, v::ExpUpFlow, op) = 
#     @inbounds @. x = op($ExpUpFlow1(x), $ExpUpFlow1(v)); nothing

# @inline accumulate(x::ExpUpFlow1, v::ExpUpFlow, w::Float64, op) = 
#     @inbounds @. x = op($ExpUpFlow1(x), $ExpUpFlow1(v) + w); nothing


# #####################
# # downward pass
# #####################

# struct ExpUpDownFlow1
#     upflow::ExpUpFlow1
#     downflow::Vector{Float64}
#     ExpUpDownFlow1(upf::ExpUpFlow1) = new(upf, log.(zeros(Float64, length(upf)) .+ 1e-300))
# end

# const ExpUpDownFlow2 = ExpUpFlow2

# const ExpUpDownFlow = Union{ExpUpDownFlow1, ExpUpDownFlow2}


# function compute_exp_flows(circuit::ProbCircuit, data)

#     # upward pass
#     @inline upflow!(n, v) = begin
#         n.data = (v isa ExpUpFlow1) ? ExpUpDownFlow1(v) : v
#         v
#     end

#     @inline upflow(n) = begin
#         d = n.data::ExpUpDownFlow
#         (d isa ExpUpDownFlow1) ? d.upflow : d
#     end

#     evaluate_exp(circuit, data; nload=upflow, nsave=upflow!, reset=false)
    
#     # downward pass

#     @inline downflow(n) = (n.data::ExpUpDownFlow1).downflow
#     @inline isfactorized(n) = n.data::ExpUpDownFlow isa ExpUpDownFlow2

#     downflow(circuit) .= 0.0

#     foreach_down(circuit; setcounter=false) do n
#         if isinner(n) && !isfactorized(n)
#             downflow_n = downflow(n)
#             upflow_n = upflow(n)
#             for ite in 1 : num_children(n)
#                 c = children(n)[ite]
#                 log_theta = is⋀gate(n) ? 0.0 : n.log_probs[ite]
#                 if isfactorized(c)
#                     upflow2_c = c.data::ExpUpDownFlow2
#                     # propagate one level further down
#                     for i = 1:2
#                         downflow_c = downflow(@inbounds children(c)[i])
#                         accumulate(downflow_c, downflow_n .+ log_theta .+ upflow2_c.prime_flow 
#                         .+ upflow2_c.sub_flow .- upflow_n, logsumexp)
#                     end
#                 else
#                     upflow1_c = (c.data::ExpUpDownFlow1).upflow
#                     downflow_c = downflow(c)
#                     accumulate(downflow_c, downflow_n .+ log_theta .+ upflow1_c .- upflow_n, logsumexp)
#                 end
#             end 
#         end
#         nothing
#     end
#     nothing
# end


# """
# Get upflow of a probabilistic circuit
# """
# @inline get_exp_upflow(pc::ProbCircuit) = get_exp_upflow(pc.data)
# @inline get_exp_upflow(elems::ExpUpDownFlow1) = elems.upflow
# @inline get_exp_upflow(elems::ExpUpFlow) = ExpUpFlow1(elems)

# """
# Get the node/edge downflow from probabilistic circuit
# """
# function get_exp_downflow(n::ProbCircuit; root=nothing)::Vector{Float64}
#     downflow(x::ExpUpDownFlow1) = x.downflow
#     downflow(x::ExpUpDownFlow2) = begin
#         ors = or_nodes(root)
#         p = findall(p -> n in children(p), ors)
#         @assert length(p) == 1
#         get_exp_downflow(ors[p[1]], n)
#     end
#     downflow(n.data)
# end

# function get_exp_downflow(n::ProbCircuit, c::ProbCircuit)::Vector{Float64}
#     @assert !is⋁gate(c) && is⋁gate(n) && c in children(n)
#     log_theta = n.log_probs[findfirst(x -> x == c, children(n))]
#     return get_exp_downflow(n) .+ log_theta .+ get_exp_upflow(c) .- get_exp_upflow(n)
# end