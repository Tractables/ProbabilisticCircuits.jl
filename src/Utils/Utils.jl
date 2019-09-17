"""
Module with general utilities and missing standard library features that could be useful in any Julia project
"""
module Utils
using StatsFuns
import StatsFuns.logsumexp

export copy_with_eltype, issomething, flatmap, map_something, ntimes, some_vector,
assign, accumulate_val, accumulate_prod, accumulate_prod_normalized, assign_prod,
assign_prod_normalized, prod_fast, count_conjunction, sum_weighted_product, 
order_asc, to_long_mi, @no_error, disjoint, typejoin, lower_element_type, map_values, groupby,
unzip, @printlog

function __init__()
    set_zero_subnormals(true) # this is supposed to speed up floating point arithmetic on certain architectures
end

# various utilities

"""
Copy the array while changing the element type
"""
copy_with_eltype(input, Eltype) = copyto!(similar(input, Eltype), input)

"""
Is the argument not `nothing`?
"""
issomething(x) = !isnothing(x)

@inline order_asc(x, y) = x > y ? (y, x) : (x , y)

function to_long_mi(m::Matrix{Float64}, min_int, max_int)::Matrix{Int64}
    δmi = maximum(m) - minimum(m)
    δint = max_int - min_int
    return @. round(Int64, m * δint / δmi + min_int)
end

macro no_error(ex)
    quote
        try
            $(esc(ex))
            true
        catch
            false
        end
    end
end

function disjoint(set1::AbstractSet, sets::AbstractSet...)::Bool
    seen = set1
    for set in sets
        if !isempty(intersect(seen,set))
            return false
        else
            seen = union(seen, set)
        end
    end
    return true
end


import Base.typejoin
"Get the most specific type parameter possible for an array"
typejoin(array) = mapreduce(e -> typeof(e), typejoin, array)

"Specialize the type parameter of an array to be most specific"
lower_element_type(array) = copy_with_eltype(array, typejoin(array))


# functional programming basics

#####################
# functional programming
#####################

# Your regular flatmap
# if you want the return array to have the right element type, provide an init with the desired type. Otherwise it may become Array{Any}
@inline flatmap(f, arr::AbstractVector, init=[]) = mapreduce(f, append!, arr; init=init)

@inline map_something(f,v) = (v == nothing) ? nothing : f(v)

ntimes(f,n) = (for i in 1:n-1; f(); end; f())

function map_values(f::Function, dict::AbstractDict{K}, vtype::Type)::AbstractDict{K,vtype} where K
    mapped_dict = Dict{K,vtype}()
    for key in keys(dict)
        mapped_dict[key] = f(dict[key])
    end
    mapped_dict
end

function groupby(f::Function, list)
    groups = Dict()
    for v in list
        push!(get!(groups, f(v), []), v)
    end
    groups
end

#####################
# probability semantics for various data types
#####################

@inline always(::Type{T}, dims::Int...) where T<:Number = ones(T, dims...)
@inline always(::Type{T}, dims::Int...) where T<:Bool = trues(dims...)

@inline never(::Type{T}, dims::Int...) where T<:Number = zeros(T, dims...)
@inline never(::Type{T}, dims::Int...) where T<:Bool = falses(dims...)

@inline some_vector(::Type{T}, dims::Int...) where T<:Number = Vector{T}(undef, dims...)
@inline some_vector(::Type{T}, dims::Int...) where T<:Bool = BitArray(undef, dims...)


#####################
# helper arithmetic to fuse complex high-arity operation
#####################

@inline safe_div(x::T, y::T) where T = iszero(y) ? 0 : x/y # seems to be much slower than replacing NaN after the fact

@inline replace_nan(x::T) where T = isnan(x) ? zero(T) : x

# need to define this because Julia says true + true = 2 instead of true
Base.@pure accumulator_op(::Type{<:Number}) = :(@fastmath +)
Base.@pure accumulator_op(::Type{Bool}) = :(|)
Base.@pure product_op(::Type{<:Number}) = :(@fastmath *)
Base.@pure product_op(::Type{Bool}) = :(&)

@inline assign(acc::AbstractArray, pr::AbstractArray) =
    acc .= pr # seems to be just as fast as copyto!

@inline @generated accumulate_val(acc::AbstractArray, x::AbstractArray) =
    :(@fastmath  acc .= $(accumulator_op(eltype(acc))).(acc, x))


const max_unroll_products = 15

# generate an expression for the loop-unrolled product of x1 and xs...
function expand_product(i, ::Type{Et}, x1::Union{Symbol,Expr}, xs::Union{Symbol,Expr}) where Et
    if i == 0
        x1
    else
        @assert i <= max_unroll_products "Unrolling loop too many times ($i), the compiler will choke... implement a loopy version instead."
        #strangely adding @inbounds here slows things down a lot
        :($(product_op(Et)).($(expand_product(i-1,Et,x1,xs)),xs[$i]))
    end
end


# Many implementations of `accumulate_prod` are very slow, and somehow don't properly fuse broadcasts':
# - acc .|= (&).(xs...) is terrible!
# - acc .|= unrolled_reduce((x,y) -> x .& y, x1, xs) is also bad, it still does memory allocations!
# Hence, meta-programming to the rescue.
@inline @generated function accumulate_prod_unroll(acc::AbstractArray{<:Number}, x1::AbstractArray{<:Number}, xs::AbstractArray{<:Number}...)
    :(@fastmath acc .= $(accumulator_op(eltype(acc))).(acc, $(expand_product(length(xs),eltype(acc),:x1,:xs))))
end

@inline function accumulate_prod(acc::AbstractArray{<:Number}, xs::AbstractVector{<:AbstractArray{<:Number}})
    if length(xs) > max_unroll_products
        tmp = prod_fast(xs[max_unroll_products:end])
        accumulate_prod_unroll(acc, tmp, xs[1:max_unroll_products-1]...)
    else
        accumulate_prod_unroll(acc, xs...)
    end
end

@inline @generated function accumulate_prod_normalized_unroll(acc::AbstractArray{<:Number}, z::AbstractArray{<:Number},
                                                        x1::AbstractArray{<:Number}, xs::AbstractArray{<:Number}...)
    x1ze = (eltype(z) <: Bool) ? :x1 : :(replace_nan.(x1 ./ z)) # Bool z is always equal to 0 or 1, and hence does nothing useful for flows (assuming z > x1)
    # use fastmath for all but the division where we require proper NaN handling
    :(acc .= $(accumulator_op(eltype(acc))).(acc, $(expand_product(length(xs),eltype(acc),x1ze,:xs))))
end

@inline function accumulate_prod_normalized(acc::AbstractArray{<:Number}, z::AbstractArray{<:Number},
                                        x1::AbstractArray{<:Number}, xs::AbstractArray{<:AbstractArray{<:Number}})
    if length(xs) > max_unroll_products
        tmp = prod_fast(xs[max_unroll_products:end])
        accumulate_prod_normalized_unroll(acc, z , x1, tmp, xs[1:max_unroll_products-1]...)
    else
        accumulate_prod_normalized_unroll(acc, z , x1, xs...)
    end
end

@inline @generated function assign_prod_unroll(acc::AbstractArray{<:Number}, x1::AbstractArray{<:Number}, xs::AbstractArray{<:Number}...)
    :(acc .= $(expand_product(length(xs),eltype(acc),:x1,:xs)))
end

@inline function assign_prod(acc::AbstractArray{<:Number}, xs::AbstractVector{<:AbstractArray{<:Number}})
    if length(xs) > max_unroll_products
        tmp = prod_fast(xs[max_unroll_products:end])
        assign_prod_unroll(acc, tmp, xs[1:max_unroll_products-1]...)
    else
        assign_prod_unroll(acc, xs...)
    end
end

@inline @generated function assign_prod_normalized_unroll(acc::AbstractArray{<:Number}, z::AbstractArray{<:Number},
                                                    x1::AbstractArray{<:Number}, xs::AbstractArray{<:Number}...)
    x1ze = (eltype(z) <: Bool) ? :x1 : :(replace_nan.(x1 ./ z)) # Bool z is always equal to 0 or 1, and hence does nothing useful for flows (assuming z > x1)
    # use fastmath for all but the division where we require proper NaN handling
    :(acc .= $(expand_product(length(xs),eltype(acc),x1ze,:xs)))
end

@inline function assign_prod_normalized(acc::AbstractArray{<:Number}, z::AbstractArray{<:Number},
                                        x1::AbstractArray{<:Number}, xs::AbstractArray{<:AbstractArray{<:Number}})
    if length(xs) > max_unroll_products
        tmp = prod_fast(xs[max_unroll_products:end])
        assign_prod_normalized_unroll(acc, z , x1, tmp, xs[1:max_unroll_products-1]...)
    else
        assign_prod_normalized_unroll(acc, z , x1, xs...)
    end
end


@inline @generated function prod_fast_unroll(x1::AbstractArray{<:Number}, xs::AbstractArray{<:Number}...)
    :(@fastmath $(expand_product(length(xs),eltype(x1),:x1,:xs)))
end

@inline function prod_fast(xs::AbstractArray{<:AbstractArray{<:Number}})
    if length(xs) > max_unroll_products
        tmp = prod_fast(xs[max_unroll_products:end])
        prod_fast_unroll(tmp, xs[1:max_unroll_products-1]...)
    else
        prod_fast_unroll(xs...)
    end
end

@inline function prod_fast(x1::AbstractArray{<:Number}, xs::AbstractArray{<:AbstractArray{<:Number}})
    if length(xs) > max_unroll_products-1
        tmp = prod_fast(xs[max_unroll_products-1:end])
        prod_fast_unroll(x1, tmp, xs[1:max_unroll_products-2]...)
    else
        prod_fast_unroll(x1, xs...)
    end
end


# Specialized version of sum_of_products for each 2-layer subcircuit
# Adds a lot of compile time but no noticable speedup over the 1-layer operators above
# test: spe3(Float16,Tuple{NTuple{3,Float16},NTuple{2,Float16},NTuple{2,Float16}})
@generated function sum_of_products(y, ps::Tuple)
    sum = expand_product(length(ps.parameters[1].parameters), Float32, :(ps[1]))
    for i in 2:length(ps.parameters)
        prod = expand_product(length(ps.parameters[i].parameters), Float32, :(ps[$i]))
        sum = :($(accumulator_op(eltype(y))).($sum, $prod))
    end
    # println(:(y .= $sum))
    :(y .= $sum)
end

function expand_product(i, et, xs::Union{Symbol,Expr})
    if i<=0
        :()
    elseif i==1
        :($xs[1])
    else
        :($(product_op(et)).($(expand_product(i-1,et,xs)),$xs[$i])) #strangely adding @inbounds here slows things down a lot
    end
end

#TODO: make this method rebust to very large xs sets as above
@inline @generated function count_conjunction(x1::BitVector, xs::BitVector...)::UInt32
    :(count($(expand_product(length(xs),eltype(x1),:x1,:xs))))
end

#TODO: make this method rebust to very large xs sets as above
# @inline @generated function sum_weighted_product(weights::AbstractArray{<:Number}, x1::AbstractArray{<:Number}, xs::AbstractArray{<:Number}...)
#     :(sum($(expand_product(length(xs),eltype(x1),:x1,:xs)) .* weights))
# end
@inline @generated function sum_weighted_product(weights::AbstractArray{<:Number}, x1::AbstractArray{<:Number}, xs::AbstractArray{<:Number}...)
    :(sum(weights[$(expand_product(length(xs),eltype(x1),:x1,:xs))]))
end


function logsumexp(A::AbstractArray, dims)
    return dropdims(mapslices(StatsFuns.logsumexp, A, dims=dims), dims=dims)
end

@inline unzip(x) = zip(x...)

using Suppressor:@suppress_err
macro printlog(filename)
    @eval begin
        @suppress_err Base.println(xs...) =
            open(f -> (println(f, xs...); println(stdout, xs...)), $filename, "a")
        @suppress_err Base.print(xs...) =
            open(f -> (print(f, xs...); print(stdout, xs...)), $filename, "a")
        #content = @suppress_err Base.@time = 
        #    open(f -> (print(f, ); @time ), $filename, "a")
    end
    nothing
end

end #module
