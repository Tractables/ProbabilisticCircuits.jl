using CUDA

#####
abstract type InputDist end
struct Indicator{T} <: InputDist
    value::T
end
const Literal = Indicator{Bool}

struct Categorical <: InputDist
    categories::UInt32
end

######
abstract type BitsNode end
struct BitsSum <: BitsNode end
struct BitsInput{D <: InputDist} <: BitsNode
    variable::Int
    dist::D
end
dist(n::BitsInput) = n.dist
dist(n::BitsNode) = nothing
########
function kernel(ans, vec, ids)
    for idx = 1:2
        cur_id = ids[idx]
        # @cuprintln(cur_id)
        item = vec[cur_id]
        d = dist(item)::InputDist
        var = item.variable
        
        CUDA.@atomic ans[ans_idx(d)] += 1
    end
    return nothing
end

ans_idx(n::Indicator{Bool}) = 1
ans_idx(n::Indicator{UInt32}) = 2
ans_idx(n::Categorical) = 3

########

function run()
    vec = Vector{Union{BitsSum, 
                       BitsInput{Indicator{Bool}}, 
                       BitsInput{Indicator{UInt32}},
                       BitsInput{Categorical}
                       }}()
    input_node_ids = [1, 2]
    push!(vec, BitsInput(1, Indicator{Bool}(false)))
    push!(vec, BitsInput(2, Indicator{UInt32}(1)))
    push!(vec, BitsInput(2, Categorical(10)))
    push!(vec, BitsSum())

    cuvec = cu(vec)
    ids = cu(input_node_ids)
    ans = CUDA.zeros(Int64, 3)

    @device_code_warntype @cuda kernel(ans, cuvec, ids)
    println(ans)
end

run()