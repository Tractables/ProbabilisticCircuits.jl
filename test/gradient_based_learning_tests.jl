using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame
using CUDA: CUDA
using Random

@testset "Gradient-based learning tests" begin
    dfb = DataFrame(BitMatrix([true true; false false]), :auto)
    weights = DataFrame(weight = [2.0, 2.0])
    wdfb = hcat(dfb, weights)
    r = fully_factorized_circuit(ProbCircuit,num_features(dfb))
    Random.seed!(123)
    uniform_parameters!(r; perturbation = 0.2)
    wdfb = batch(wdfb, 2);
    
    params = nothing
    for i = 1 : 200
        params = sgd_parameter_learning(r, wdfb; lr = 0.01)
    end
    @test exp.(params) ≈ [0.5, 0.5, 0.5, 0.5, 1.0] atol = 0.1
    
    if CUDA.functional()
        params = nothing
        for i = 1 : 200
            params = sgd_parameter_learning(r, wdfb; lr = 0.01)
        end
        @test to_cpu(exp.(params)) ≈ [0.5, 0.5, 0.5, 0.5, 1.0] atol = 0.1
    end
end