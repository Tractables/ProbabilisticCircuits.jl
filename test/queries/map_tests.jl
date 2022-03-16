using Test
using ProbabilisticCircuits
using ProbabilisticCircuits: CuBitsProbCircuit
using CUDA
import Distributions

include("../helper/data.jl")
include("../helper/plain_dummy_circuits.jl")

@testset "MAP regression test" begin
    a,b    = [ProbabilisticCircuits.PlainInputNode(i, Indicator(true)) for i=1:2]
    a_, b_ = [ProbabilisticCircuits.PlainInputNode(i, Indicator(false)) for i=1:2]
    circuit = 0.6 * (a * (.5 * b + .5 * b_)) + .4 * (a_ * (0.9 * b + .1 * b_))
    no_data = Matrix{Union{Bool,Missing}}([missing missing])
    maps, mappr = MAP(circuit, no_data; batch_size = 1, return_map_prob=true)
    @test mappr[1] ≈ log(0.4 * 0.9)
    @test maps[1,1] == false && maps[1,2] == true
    
    complete_states = Matrix([true true; true false; false true; false false])
    mar = loglikelihoods(circuit, complete_states; batch_size = 3)
    @test all(mappr .> mar .- 1e-6)

    if CUDA.functional()
        bpc = CuBitsProbCircuit(circuit)
        no_data_gpu = cu(no_data)
        maps_gpu = MAP(bpc, no_data_gpu; batch_size=1)

        @test Matrix(maps_gpu) == maps 
    end
end

@testset "MAP" begin
    prob_circuit = little_4var()
    
    # A. Full Data
    data_full = generate_data_all(num_randvars(prob_circuit))
    if CUDA.functional()
        bpc = CuBitsProbCircuit(prob_circuit)
        data_full_gpu = cu(data_full)
    end

    map, mappr = MAP(prob_circuit, data_full; batch_size = 1, return_map_prob=true)
    @test map == data_full
    evipr = loglikelihoods(prob_circuit, data_full; batch_size = 16)
    @test mappr ≈ evipr atol=1e-6

    if CUDA.functional()
        maps_gpu = MAP(bpc, data_full_gpu; batch_size=1)
        @test Matrix(maps_gpu) == map 
    end

    # B. Partial Data; test if non-missing MAP values are same as data (as they should be)
    data_marg = Matrix([false false false false; 
                      false true true false; 
                      false false true true;
                      false false false missing; 
                      missing true false missing; 
                      missing missing missing missing; 
                      false missing missing missing])

    map, mappr = MAP(prob_circuit, data_marg; batch_size = 1, return_map_prob=true)

    @test all(ismissing.(data_marg) .| (data_marg .== map))
    mar = loglikelihoods(prob_circuit, data_marg; batch_size = 16)
    @test all(mar .> mappr .- 1e-6)

    if CUDA.functional()
        data_marg_gpu = cu(data_marg)

        # bigger batch size
        maps_gpu = MAP(bpc, data_marg_gpu; batch_size=16)
        @test Matrix(maps_gpu) == map

        # smaller batch size
        maps_gpu = MAP(bpc, data_marg_gpu; batch_size=2)
        @test Matrix(maps_gpu) == map
    end

    # C. Check specific MAP queries with known result
    data_c = Matrix([false false false missing])
    true_map = Matrix([false false false true])
    map, mappr = MAP(prob_circuit, data_c; batch_size = 1, return_map_prob=true)
    @test map == true_map
    @test mappr[1] ≈ -1.2729657

    if CUDA.functional()
        data_c_gpu = cu(data_c)
        maps_gpu = MAP(bpc, data_c_gpu; batch_size=1)
        @test Matrix(maps_gpu) == true_map 
    end

    # D. TODO. Add tests with different input types for map
    #          Generate all possible missing patches and compute map on cpu vs gpu
end


@testset "Binomial MAP Test" begin
    EPS = 1e-6
    EPS2 = 1e-3

    # p = 0.0
    pc = InputNode(1, Binomial(5, 0.0));
    data = Matrix(transpose([missing;; UInt32(3)]))
    true_map = [UInt32(0), UInt32(3)]
    our_map = MAP(pc, data; batch_size=2);
    @test all( true_map .== our_map )

    # p = 1.0
    pc = InputNode(1, Binomial(5, 1.0));
    true_map = [UInt32(5), UInt32(3)];
    our_map = MAP(pc, data; batch_size=2);
    @test all( true_map .== our_map )

    # p = 0.4
    N = 10
    p  = 0.4
    pc = InputNode(1, Binomial(N, p));
    true_map = [UInt32(4), UInt32(3)];
    our_map = MAP(pc, data; batch_size=2);
    @test all( true_map .== our_map )
    

    if CUDA.functional()
        pc2 = summate([pc])
        bpc = CuBitsProbCircuit(pc2)
        cu_data = cu(data)

        our_map = Array(MAP(bpc, cu_data; batch_size=2))
        @test all( true_map .== our_map )
    end
end
