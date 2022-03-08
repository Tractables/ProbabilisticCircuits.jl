using CUDA

#####
abstract type InputDist end
struct Indicator{T} <: InputDist
    value::T
end

struct IndicatorBool <:InputDist end
struct IndicatorUInt32 <:InputDist end

struct Categorical <: InputDist end
struct Binomial <: InputDist end
######
abstract type BitsNode end
struct BitsSum <: BitsNode end
struct BitsInput{D <: InputDist} <: BitsNode
    dist::D
end
dist(n::BitsInput) = n.dist
dist(n::BitsNode) = nothing
########
ans_idx(n) = 1
ans_idx(n::Indicator{Bool}) = 2
ans_idx(n::Indicator{UInt32}) = 3
ans_idx(n::Indicator{Int64}) = 4
ans_idx(n::Categorical) = 5
ans_idx(n::Binomial) = 6
ans_idx(n::IndicatorBool) = 7
ans_idx(n::IndicatorUInt32) = 8

########
function kernel(ans, vec, ids)
    for idx = 1:size(ids, 1)
        cur_id = ids[idx]
        item = vec[cur_id]::BitsInput
        d = dist(item)
        i = ans_idx(d)::Int64
        ans[i] += 1
    end
    return nothing
end

########

function vec_A()
    vec = Vector{Union{
        BitsInput{Indicator{Bool}}, 
        BitsInput{Indicator{UInt32}},
        BitsInput{Categorical},
        #    BitsSum
    }}()

    push!(vec, BitsInput(Indicator{Bool}(false)))
    push!(vec, BitsInput(Indicator{UInt32}(1)))
    push!(vec, BitsInput(Categorical()))
    # push!(vec, BitsSum())
    return vec
end

function vec_B()
    vec = Vector{Union{
        BitsInput{Categorical},
        BitsInput{Binomial},
        BitsSum
    }}()
    push!(vec, BitsInput(Categorical()))
    push!(vec, BitsInput(Binomial()))
    push!(vec, BitsSum())
    return vec
end

function vec_C()
    vec = Vector{Union{
        BitsInput{Indicator{Bool}}, 
        BitsInput{Indicator{UInt32}},
        BitsInput{Indicator{Int64}},
        # BitsInput{Categorical},
        # BitsInput{Binomial},
        # BitsSum
    }}()

    push!(vec, BitsInput(Indicator{Bool}(false)))
    push!(vec, BitsInput(Indicator{UInt32}(UInt32(1))))
    # push!(vec, BitsInput(Indicator{Int64}(1)))
    # push!(vec, BitsSum())
    return vec
end

function vec_D()
    vec = Vector{Union{ 
        BitsInput{Indicator{UInt32}},
        BitsInput{Categorical},
        BitsInput{Binomial},
        BitsSum
    }}()

    push!(vec, BitsInput(Categorical()))
    push!(vec, BitsInput(Binomial()))
    push!(vec, BitsSum())
    return vec
end

function vec_E()
    vec = Vector{Union{
        BitsInput{Indicator{UInt32}},
        BitsInput{Categorical},
        BitsInput{Binomial},
        BitsSum
    }}()

    push!(vec, BitsInput(Categorical()))
    push!(vec, BitsInput(Binomial()))
    push!(vec, BitsSum())
    return vec
end

function run(get_vec)
    vec = get_vec()
    input_node_ids = Vector{UInt32}()
    for i = 1:length(vec)
        if vec[i] isa BitsInput
            append!(input_node_ids, UInt32(i))
        end
    end
    ans = CUDA.zeros(Int64, 10)
    @cuda kernel(ans, cu(vec), cu(input_node_ids))
    println(ans)
end



run(vec_A)  # Runs fine
run(vec_B)  # Runs fine
run(vec_C)  # Runs fine
# run(vec_D) # Fails 
run(vec_E) # Fails

