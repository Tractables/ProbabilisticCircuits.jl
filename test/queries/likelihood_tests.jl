using Test, DirectedAcyclicGraphs, ProbabilisticCircuits, CUDA
using ProbabilisticCircuits: CuProbBitCircuit

include("../helper/plain_dummy_circuits.jl")

@testset "likelihood" begin
    
    pc = little_3var()
    bpc = CuProbBitCircuit(pc)

    data = cu([true true false; false true false; false false false])

    lls = Array(loglikelihoods(bpc, data; batch_size = 32))
    avg_ll = loglikelihood(bpc, data; batch_size = 32)
    
    @test lls[1] ≈ log(Float32(0.125))
    @test lls[2] ≈ log(Float32(0.125))
    @test lls[3] ≈ log(Float32(0.125))
    @test avg_ll ≈ log(Float32(0.125))

    pc = little_3var_bernoulli(; p = Float32(0.6))
    bpc = CuProbBitCircuit(pc)

    lls = Array(loglikelihoods(bpc, data; batch_size = 32))

    @test lls[1] ≈ log(Float32(0.6 * 0.6 * 0.4))
    @test lls[2] ≈ log(Float32(0.4 * 0.6 * 0.4))
    @test lls[3] ≈ log(Float32(0.4 * 0.4 * 0.4))

    data = cu(UInt32.([2 3 4; 5 1 2; 3 4 5]))

    pc = little_3var_categorical(; num_cats = UInt32(5))
    bpc = CuProbBitCircuit(pc)

    lls = Array(loglikelihoods(bpc, data; batch_size = 32))

    @test lls[1] ≈ log(Float32(0.2^3))

end