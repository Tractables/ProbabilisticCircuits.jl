using Test, ProbabilisticCircuits
using ProbabilisticCircuits: PsddFormat

include("../helper/plain_dummy_circuits.jl")
include("../helper/pc_equals.jl")

@testset "Psdd IO tests" begin
    
    # Indicators
    pc = little_3var()

    mktempdir() do tmp
        file = "$tmp/example.psdd"
        write(file, pc)
        # note: number of nodes can changes because of "true" leafs
        pc2 = read(file, ProbCircuit)
        write(file, pc2)
        pc3 = read(file, ProbCircuit)

        test_pc_equals(pc2, pc3)

        file = "$tmp/example.psdd.gz"
        write(file, pc2)
        pc3 = read(file, ProbCircuit)

        test_pc_equals(pc2, pc3)
    end

end
