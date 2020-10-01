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

@testset "EM tests" begin
    data = DataFrame([true missing])
    vtree2 = PlainVtree(2, :balanced)
    pc = fully_factorized_circuit(StructProbCircuit, vtree2).children[1]
    uniform_parameters(pc)
    pc.children[1].prime.log_probs .= log.([0.3, 0.7])
    pc.children[1].sub.log_probs .= log.([0.4, 0.6])
    pbc = ParamBitCircuit(pc, data)
    estimate_parameters_em(pc, data; pseudocount=0.0)
    @test all(pc.children[1].prime.log_probs .== log.([1.0, 0.0]))
    @test pc.children[1].sub.log_probs[1] .≈ log.([0.4, 0.6])[1] atol=1e-6
end