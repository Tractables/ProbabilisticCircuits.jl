using Pkg; Pkg.activate(@__DIR__)

using ProbabilisticCircuits
using ProbabilisticCircuits: CuBitsProbCircuit, loglikelihood, full_batch_em, mini_batch_em
using MLDatasets
using CUDA

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


function run()
    train_gpu, test_gpu = mnist_gpu()

    latents = 120
    pseudocount = 0.01

    println("Generating HCLT structure with $latents latents... ");
    @time pc = hclt(train_gpu, latents; pseudocount, input_type = Literal);
    init_parameters(pc; perturbation = 0.4);
    println("Number of free parameters: $(num_parameters(pc))")

    print("Moving circuit to GPU... ")
    CUDA.@time bpc = CuBitsProbCircuit(pc)

    batch_size  = 512
    pseudocount = .005
    softness    = 0
    
    print("First round of minibatch EM... ")
    CUDA.@time mini_batch_em(bpc, train_gpu, 400; batch_size, pseudocount, 
    			 softness, param_inertia = 0.01, param_inertia_end = 0.95)
    			 
    CUDA.@time mini_batch_em(bpc, train_gpu, 100; batch_size, pseudocount, 
    			 softness, param_inertia = 0.95, param_inertia_end = 0.999)
    
    CUDA.@time full_batch_em(bpc, train_gpu, 10; batch_size, pseudocount, softness)

    print("Update parameters... ")
    @time ProbabilisticCircuits.update_parameters(bpc)
    pc
end


run()
