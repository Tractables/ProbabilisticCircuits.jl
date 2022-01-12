using MLDatasets, DataFrames
using ProbabilisticCircuits
using CUDA

train_int = transpose(reshape(MNIST.traintensor(UInt8), 28*28, :));

function bitsfeatures(data_int)
    data_bits = zeros(Bool, size(data_int,1), 28*28*8)
    for ex = 1:size(data_int,1), pix = 1:size(data_int,2)
        x = data_int[ex,pix]
        for b = 0:7
            if (x & (one(UInt8) << b)) != zero(UInt8)
                data_bits[ex, (pix-1)*8+b+1] = true
            end
        end
    end
    data_bits
end

train_bits = bitsfeatures(train_int);

train_bits_gpu = cu(train_bits);

train_bits_gpu2 = train_bits_gpu .+ 1
weights = CUDA.ones(Float32,size(train_bits_gpu2,1));
@time CUDA.@sync pairwise_marginals(train_bits_gpu2, weights)

function pairwise_counts(data)
    not_data = .!data
    count11 = transpose(data) * data
    count10 = transpose(not_data) * data
    count01 = transpose(data) * not_data
    count00 = size(data,1) .- count11 .- count10 .- count01
    count1 = reduce(+, data; dims=1)
    count0 = size(data,1) .- count1
    count11, count10, count01, count00, count1, count0
end

function mutual_info(data; pseudocount=1.0)
    count11, count10, count01, count00, count1, count0 = pairwise_counts(data)
    mi  = count11 .* log.(count11 ./ (transpose(count1) * count1))
    mi += count10 .* log.(count10 ./ (transpose(count1) * count0))
    mi += count01 .* log.(count01 ./ (transpose(count0) * count1))
    mi += count00 .* log.(count00 ./ (transpose(count0) * count0))
    mi
    # TODO pseudocount
end

# @time CUDA.@sync pairwise_counts(train_bits_gpu);
@time CUDA.@sync mutual_info(train_bits_gpu);
