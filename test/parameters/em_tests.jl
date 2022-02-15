using Test, DirectedAcyclicGraphs, ProbabilisticCircuits
using ProbabilisticCircuits: PlainSumNode, PlainMulNode, PlainProbCircuit
using CUDA
import ProbabilisticCircuits as PCs

include("../helper/plain_dummy_circuits.jl")


@testset "mini-batch em" begin

    # LiteralDist
    
    pc = little_3var()
    bpc = PCs.CuProbBitCircuit(pc)

    data = cu([true true false; false true false; false false false])

    lls = mini_batch_em(bpc, data, 2; batch_size = 3, pseudocount = 0.1, param_inertia = 0.2, verbose = false)

    @test lls[2] > lls[1]

    # BernoulliDist

    pc = little_3var_bernoulli()
    bpc = PCs.CuProbBitCircuit(pc)

    data = cu([true true false; false true false; false false false])

    lls = mini_batch_em(bpc, data, 2; batch_size = 3, pseudocount = 0.1, param_inertia = 0.2, verbose = false)

    @test lls[2] > lls[1]

    # CategoricalDist

    pc = little_3var_categorical(; num_cats = UInt32(5))
    bpc = PCs.CuProbBitCircuit(pc)

    data = cu(UInt32.([2 3 4; 5 1 2; 3 4 5]))

    lls = mini_batch_em(bpc, data, 2; batch_size = 3, pseudocount = 0.1, param_inertia = 0.2, verbose = false)

    @test lls[2] > lls[1]

end

@testset "full-batch em" begin

    # LiteralDist
    
    pc = little_3var()
    bpc = PCs.CuProbBitCircuit(pc)

    data = cu([true true false; false true false; false false false])

    lls = full_batch_em(bpc, data, 2; batch_size = 32, pseudocount = 0.1, verbose = false)

    @test lls[2] > lls[1]

    # BernoulliDist

    pc = little_3var_bernoulli()
    bpc = PCs.CuProbBitCircuit(pc)

    data = cu([true true false; false true false; false false false])

    lls = full_batch_em(bpc, data, 2; batch_size = 32, pseudocount = 0.1, verbose = false)

    @test lls[2] > lls[1]

    # CategoricalDist

    pc = little_3var_categorical(; num_cats = UInt32(5))
    bpc = PCs.CuProbBitCircuit(pc)

    data = cu(UInt32.([2 3 4; 5 1 2; 3 4 5]))

    lls = full_batch_em(bpc, data, 2; batch_size = 32, pseudocount = 0.1, verbose = false)

    @test lls[2] > lls[1]

end

@testset "em with missing" begin

    # LiteralDist
    
    pc = little_3var()
    bpc = PCs.CuProbBitCircuit(pc)

    data = cu([true true missing; false missing false; false false false])

    lls = full_batch_em(bpc, data, 2; batch_size = 32, pseudocount = 0.1, verbose = false)

    @test lls[2] > lls[1]

    # BernoulliDist

    pc = little_3var_bernoulli()
    bpc = PCs.CuProbBitCircuit(pc)

    data = cu([true missing false; missing true false; false false false])

    lls = full_batch_em(bpc, data, 2; batch_size = 32, pseudocount = 0.1, verbose = false)

    @test lls[2] > lls[1]

    # CategoricalDist

    pc = little_3var_categorical(; num_cats = UInt32(5))
    bpc = PCs.CuProbBitCircuit(pc)

    data = cu([missing 3 4; 5 1 missing; 3 4 5])

    lls = full_batch_em(bpc, data, 2; batch_size = 32, pseudocount = 0.1, verbose = false)

    @test lls[2] > lls[1]

end