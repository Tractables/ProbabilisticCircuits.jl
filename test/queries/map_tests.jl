using Test
using ProbabilisticCircuits
using CUDA

include("../helper/gpu.jl")
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
end

@testset "MAP" begin
    prob_circuit = little_4var()
    data_full = generate_data_all(num_randvars(prob_circuit))

    map, mappr = MAP(prob_circuit, data_full; batch_size = 1, return_map_prob=true)
    @test map == data_full

    evipr = loglikelihoods(prob_circuit, data_full; batch_size = 16)
    @test mappr ≈ evipr atol=1e-6
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

    map, mappr = MAP(prob_circuit, Matrix([false false false missing]); batch_size = 1, return_map_prob=true)
    @test map == Matrix([false false false true])
    @test mappr[1] ≈ -1.2729657
end