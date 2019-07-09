module Utils

export copy_with_eltype, issomething, flatmap, map_something, ntimes, some_vector,
assign, accumulate_val, accumulate_prod, accumulate_prod_normalized, assign_prod,
assign_prod_normalized, prod_fast, count_conjunction

# various utilities

copy_with_eltype(input, Eltype) = copyto!(similar(input, Eltype), input)

set_zero_subnormals(true) # this is supposed to speed up floating point arithmetic on certain architectures

issomething(x) = !isnothing(x)

# functional programming basics

#####################
# functional programming
#####################

# Your regular flatmap
# if you want the return array to have the right element type, provide an init with the desired type. Otherwise it may become Array{Any}
@inline flatmap(f, arr::AbstractVector, init=[]) = mapreduce(f, append!, arr; init=init)

@inline map_something(f,v) = (v == nothing) ? nothing : f(v)

ntimes(f,n) = (for i in 1:n-1; f(); end; f())

@inline safe_div(x::T, y::T) where T = iszero(y) ? 0 : x/y # seems to be much slower than replacing NaN after the fact

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

@inline @generated function count_conjunction(x1::BitVector, xs::BitVector...)::UInt32
    :(count($(expand_product(length(xs),eltype(x1),:x1,:xs))))
end

@inline @generated function sum_weighted_conjunction(weights::AbstractVector{W}, x1::BitVector, xs::BitVector...)::W where W
    :(sum($(expand_product(length(xs),eltype(x1),:x1,:xs)) .* weights))
end

end #module