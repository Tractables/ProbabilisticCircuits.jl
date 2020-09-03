using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame
using CUDA: CUDA

@testset "MLE tests" begin
    
    dfb = DataFrame(BitMatrix([true false; true true; false true]))
    r = fully_factorized_circuit(ProbCircuit,num_features(dfb))
    
    estimate_parameters(r,dfb; pseudocount=1.0)
    @test log_likelihood_avg(r,dfb) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=1.0)

    estimate_parameters(r,dfb; pseudocount=0.0)
    @test log_likelihood_avg(r,dfb) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=0.0)

    if CUDA.functional()

        dfb_gpu = to_gpu(dfb)
        
        estimate_parameters(r,dfb_gpu; pseudocount=1.0)
        @test log_likelihood_avg(r,dfb_gpu) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=1.0)

        estimate_parameters(r,dfb_gpu; pseudocount=0.0)
        @test log_likelihood_avg(r,dfb_gpu) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=0.0)

    end

end