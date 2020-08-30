using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame

@testset "MLE tests" begin
    
    dfb = DataFrame(BitMatrix([true false; true true; false true]))
    r = fully_factorized_circuit(ProbCircuit,num_features(dfb))
    estimate_parameters(r,dfb; pseudocount=1.0)

    # TODO test identical likelihoods with
    # ll1  = LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=1)
    # ll0 = LogicCircuits.Utils.fully_factorized_log_likelihood(dfb)

end