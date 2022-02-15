using ProbabilisticCircuits
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, loglikelihoods, full_batch_em, mini_batch_em
using MLDatasets
using CUDA
using DirectedAcyclicGraphs

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
    train, test = mnist_cpu()
    train_gpu, test_gpu = mnist_gpu()

    latents = 60
    pseudocount = 0.1

    println("Generating HCLT structure with $latents latents... ");
    @time pc = hclt(train, latents; pseudocount, input_type = LiteralDist);
    init_parameters(pc; perturbation = 0.4);
    println("Number of free parameters: $(num_parameters(pc))")

    print("Moving circuit to GPU... ")
    CUDA.@time bpc = CuBitsProbCircuit(BitsProbCircuit(pc))

    batch_size  = 512
    pseudocount = 0.1
    softness    = 0
    epochs      = 100
    @time mini_batch_em(bpc, train_gpu, epochs; batch_size, pseudocount, 
    			 softness, param_inertia = 0.2, param_inertia_end = 0.9)
    			 
    @time mini_batch_em(bpc, train_gpu, 200; batch_size, pseudocount, 
    			 softness, param_inertia = 0.9, param_inertia_end = 0.95)
    
    @time full_batch_em(bpc, train_gpu, 10; batch_size, pseudocount, softness)

    print("update parameters")
    @time ProbabilisticCircuits.update_parameters(bpc)
end


run()
