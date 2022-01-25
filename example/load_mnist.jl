using MLDatasets, CUDA

function mnist_cpu()
    train_int = transpose(reshape(MNIST.traintensor(UInt8), 28*28, :));
    test_int = transpose(reshape(MNIST.testtensor(UInt8), 28*28, :));

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

    train_cpu = bitsfeatures(train_int);
    test_cpu = bitsfeatures(test_int);

    train_cpu, test_cpu
end

function mnist_gpu()
    cu.(mnist_cpu())
end