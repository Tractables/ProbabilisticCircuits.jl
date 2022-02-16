using Test, ProbabilisticCircuits
using ProbabilisticCircuits: JpcFormat

include("../helper/plain_dummy_circuits.jl")
include("../helper/pc_equals.jl")

@testset "Jpc IO tests" begin
    
    # Indicators
    pc = little_3var()

    mktempdir() do tmp
        file = "$tmp/example.jpc"
        write(file, pc)

        pc2 = read(file, ProbCircuit)
        test_pc_equals(pc, pc2)

        pc2 = read(file, ProbCircuit, JpcFormat(), true)
        test_pc_equals(pc, pc2)

        pc2 = read(file, ProbCircuit, JpcFormat(), false)
        test_pc_equals(pc, pc2)
    end

end
