using Pkg; Pkg.activate("$(@__DIR__)")

using ProbabilisticCircuits
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, loglikelihood, full_batch_em, mini_batch_em

println("Loading Data")
include("load_mnist.jl");
train, test = mnist_gpu()

function experiment(train, test, epochs1, epochs2, epochs3, latents;
    batch_size=512, pseudocount, softness, param_inertia1, param_inertia_end1, param_inertia2, param_inertia_end2, shuffle=:each_epoch)

    println("Generating HCLT structure with $latents latents")
    circuit = hclt(train; num_cats=2, num_hidden_cats = latents)
    uniform_parameters!(circuit; perturbation = 0.4)

    println("Number of free parameters: $(num_parameters(circuit))")

    println("Moving circuit to GPU")
    bpc = CuBitsProbCircuit(BitsProbCircuit(circuit))

    println("mini_batch_em(bpc, train, $epochs1; 
        batch_size=$batch_size, pseudocount=$pseudocount, softness=$softness, param_inertia=$param_inertia1, param_inertia_end=$param_inertia_end1, shuffle=$shuffle)")
    
    @time mini_batch_em(bpc, train, epochs1; 
        batch_size, pseudocount, softness, param_inertia = param_inertia1, param_inertia_end = param_inertia_end1, shuffle)

    print("Test set log likelihood is ")
    println(loglikelihood(test, bpc; batch_size))

    println("mini_batch_em(bpc, train, $epochs2; 
        batch_size=$batch_size, pseudocount=$pseudocount, softness=$softness, param_inertia=$param_inertia2, param_inertia_end=$param_inertia_end2, shuffle=$shuffle)")
    
    @time mini_batch_em(bpc, train, epochs2; 
        batch_size, pseudocount, softness, param_inertia=param_inertia2, param_inertia_end=param_inertia_end2, shuffle)

    print("Test set log likelihood is ")
    println(loglikelihood(test, bpc; batch_size))

    println("full_batch_em(bpc, train, $epochs3; 
    batch_size=$batch_size, pseudocount=$pseudocount, softness=$softness)")
    @time full_batch_em(bpc, train, epochs3; 
        batch_size, pseudocount, softness)

    print("Test set log likelihood is ")
    println(loglikelihood(test, bpc; batch_size))

    bpc
end

bpp(ll) = ll / log(2) / 28 / 28