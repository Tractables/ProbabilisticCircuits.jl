export to_long_mi,
    generate_all, generate_data_all


###################
# Misc.
####################


function to_long_mi(m::Matrix{Float64}, min_int, max_int)::Matrix{Int64}
    δmi = maximum(m) - minimum(m)
    δint = max_int - min_int
    return @. round(Int64, m * δint / δmi + min_int)
end

###################
# One-Hot Encoding
####################
"""
One-hot encode data (2-D Array) based on categories (1-D Array)
Each row of the return value is a concatenation of one-hot encoding of elements of the same row in data
Assumption: both input arrays have elements of same type
"""
function one_hot_encode(X::Array{T, 2}, categories::Array{T,1}) where {T<:Any}
    X_dash = zeros(Bool, size(X)[1], length(categories)*size(X)[2])
    for i = 1:size(X)[1], j = 1:size(X)[2]
            X_dash[i, (j-1)*length(categories) + findfirst(==(X[i,j]), categories)] = 1
    end  
    X_dash
end

###################
# Testing Utils
####################

"""
Given some missing values generates all possible fillings
"""
function generate_all(row::Array{Int8})
    miss_count = count(row .== -1)
    lits = length(row)
    result = Bool.(zeros(1 << miss_count, lits))

    if miss_count == 0
        result[1, :] = copy(row)
    else
        for mask = 0: (1<<miss_count) - 1
            cur = copy(row)
            cur[row .== -1] = transpose(parse.(Bool, split(bitstring(mask)[end-miss_count+1:end], "")))
            result[mask+1,:] = cur
        end
    end
    result
end

"""
Generates all possible binary configurations of size N
"""
function generate_data_all(N::Int)
    data_all = transpose(parse.(Bool, split(bitstring(0)[end-N+1:end], "")));
    for mask = 1: (1<<N) - 1
        data_all = vcat(data_all,
            transpose(parse.(Bool, split(bitstring(mask)[end-N+1:end], "")))
        );
    end
    data_all
end
