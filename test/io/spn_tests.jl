using Test, ProbabilisticCircuits
using ProbabilisticCircuits: PsddFormat

include("../helper/plain_dummy_circuits.jl")
include("../helper/pc_equals.jl")

@testset "Spn IO tests" begin
    
    # Indicators
    pc = little_3var()

    mktempdir() do tmp
        file = "$tmp/example.spn"
        write(file, pc)
        pc2 = read(file, ProbCircuit)

        test_pc_equals(pc, pc2)
    end

end
