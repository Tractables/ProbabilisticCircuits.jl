using ProbabilisticCircuits
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, loglikelihoods, full_batch_em, mini_batch_em
using MLDatasets
using CUDA


function mnist_cpu()
    train_cpu = collect(transpose(reshape(MNIST.traintensor(UInt8), 28*28, :)))
    test_cpu = collect(transpose(reshape(MNIST.testtensor(UInt8), 28*28, :)))
    train_cpu, test_cpu
end

function mnist_gpu()
    cu.(mnist_cpu())
end

function truncate(data::Matrix; bits)
    data .÷ 2^bits
end

function run()
    train, test = mnist_cpu()
    train_gpu, test_gpu = mnist_gpu()
    
    trunc_train = cu(truncate(train; bits = 5))

    latents = 32

    println("Generating HCLT structure with $latents latents... ");
    @time pc = hclt(trunc_train, latents; num_cats = 256, pseudocount = 0.1, input_type = CategoricalDist);
    init_parameters(pc; perturbation = 0.4);
    println("Number of free parameters: $(num_parameters(pc))")

    print("Moving circuit to GPU... ")
    CUDA.@time bpc = CuBitsProbCircuit(pc)

    batch_size  = 256
    pseudocount = 0.1
    softness    = 0
    epochs      = 100
    @time mini_batch_em(bpc, train_gpu, epochs; batch_size, pseudocount, 
    			 softness, param_inertia = 0.2, param_inertia_end = 0.9, debug = false)
    			 
    @time mini_batch_em(bpc, train_gpu, 200; batch_size, pseudocount, 
    			 softness, param_inertia = 0.9, param_inertia_end = 0.95)
    
    @time full_batch_em(bpc, train_gpu, 40; batch_size, pseudocount, softness)

    print("update parameters")
    @time ProbabilisticCircuits.update_parameters(bpc)

end


run()
