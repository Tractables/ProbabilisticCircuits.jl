using CUDA

#####
abstract type InputDist end
struct Indicator{T} <: InputDist
    value::T
end
const Literal = Indicator{Bool}
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
        if d isa Literal
            CUDA.@atomic ans[var] += 1
        end
    end
    return nothing
end

########

function run()
    vec = Vector{Union{BitsSum, BitsInput{Indicator{Bool}}, BitsInput{Indicator{UInt32}}}}()
    input_node_ids = [1, 2]
    push!(vec, BitsInput(1, Indicator{Bool}(false)))
    push!(vec, BitsInput(2, Indicator{UInt32}(1)))
    push!(vec, BitsSum())

    cuvec = cu(vec)
    ids = cu(input_node_ids)
    ans = CUDA.zeros(Int64, 2)

    @device_code_warntype @cuda kernel(ans, cuvec, ids)
    println(ans)
end

run()