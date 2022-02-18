using Distributed
using CUDA

if nprocs() - 1 < length(devices())
    addprocs(length(devices()) - nprocs() + 1)
end

@everywhere using CUDA

# assign devices
asyncmap((zip(workers(), devices()))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        device!(d)
    end
end

@everywhere using ProbabilisticCircuits
@everywhere using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, loglikelihoods, full_batch_em, mini_batch_em
@everywhere using MLDatasets


@everywhere function mnist_cpu()
    train_cpu = collect(transpose(reshape(MNIST.traintensor(UInt8), 28*28, :)))
    test_cpu = collect(transpose(reshape(MNIST.testtensor(UInt8), 28*28, :)))
    train_cpu, test_cpu
end

@everywhere function mnist_gpu()
    cu.(mnist_cpu())
end

@everywhere function truncate(data::Matrix; bits)
    data .÷ 2^bits
end

@everywhere function run(; batch_size = 512, num_epochs1 = 100, num_epochs2 = 100, num_epochs3 = 20, 
             pseudocount = 0.1, latents = 32, param_inertia1 = 0.2, param_inertia2 = 0.9, param_inertia3 = 0.95)
    train, test = mnist_cpu()
    train_gpu, test_gpu = mnist_gpu()
    
    trunc_train = cu(truncate(train; bits = 4))

    println("Generating HCLT structure with $latents latents... ");
    @time pc = hclt(trunc_train[1:5000,:], latents; num_cats = 256, pseudocount = 0.1, input_type = Categorical);
    init_parameters(pc; perturbation = 0.4);
    println("Number of free parameters: $(num_parameters(pc))")

    print("Moving circuit to GPU... ")
    CUDA.@time bpc = CuBitsProbCircuit(pc)

    softness    = 0
    @time mini_batch_em(bpc, train_gpu, num_epochs1; batch_size, pseudocount, 
    			 softness, param_inertia = param_inertia1, param_inertia_end = param_inertia2, debug = false)

    ll1 = loglikelihood(bpc, test_gpu; batch_size)
    println("test LL: $(ll1)")
    			 
    @time mini_batch_em(bpc, train_gpu, num_epochs2; batch_size, pseudocount, 
    			 softness, param_inertia = param_inertia2, param_inertia_end = param_inertia3)

    ll2 = loglikelihood(bpc, test_gpu; batch_size)
    println("test LL: $(ll2)")
    
    @time full_batch_em(bpc, train_gpu, num_epochs3; batch_size, pseudocount, softness)

    ll3 = loglikelihood(bpc, test_gpu; batch_size)
    println("test LL: $(ll3)")

    print("update parameters")
    @time ProbabilisticCircuits.update_parameters(bpc)

    ll1, ll2, ll3, batch_size, pseudocount, latents
end


experiments = begin
    exps = []
    for batch_size in [64, 256, 512]
        for pseudocount in [0.1, 0.01]
            for latents in [32, 48, 64, 128]
                for param_inertia1 in [0.2, 0.5]
                    for param_inertia2 in [0.8, 0.9]
                        for param_inertia3 in [0.95, 0.98]
                            push!(exps, (batch_size, pseudocount, latents))
                        end
                    end
                end
            end
        end
    end
    exps
end

results = pmap(experiments) do exper
    result = run(; batch_size = exper[1], pseudocount = exper[2], latents = exper[3])
    open("cat_hclt.log", "a+") do io
        write(io, "$(result)\n")
    end
    result
end

for result in results
    println(result)
end


