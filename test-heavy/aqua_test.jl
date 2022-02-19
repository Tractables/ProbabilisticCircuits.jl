using Aqua
using ProbabilisticCircuits
using Test

@testset "Aqua tests" begin
    Aqua.test_all(ProbabilisticCircuits, 
                    ambiguities = false)
end