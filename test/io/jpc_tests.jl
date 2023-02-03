using Test, ProbabilisticCircuits
using ProbabilisticCircuits: JpcFormat

include("../helper/plain_dummy_circuits.jl")
include("../helper/pc_equals.jl")

@testset "Jpc IO tests Literal" begin
    
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

        file = "$tmp/example.jpc.gz"
        write(file, pc)

        pc2 = read(file, ProbCircuit)
        test_pc_equals(pc, pc2)
        
    end

end

@testset "Jpc IO tests categorical" begin
    
    pc = little_3var_categorical()

    mktempdir() do tmp
        
        file = "$tmp/example.jpc"
        write(file, pc)

        pc2 = read(file, ProbCircuit)
        test_pc_equals(pc, pc2)

        pc2 = read(file, ProbCircuit, JpcFormat(), true)
        test_pc_equals(pc, pc2)

        pc2 = read(file, ProbCircuit, JpcFormat(), false)
        test_pc_equals(pc, pc2)

        file = "$tmp/example.jpc.gz"
        write(file, pc)

        pc2 = read(file, ProbCircuit)
        test_pc_equals(pc, pc2)
        
    end

end

@testset "JPC IO tests Binomial" begin
    pc = little_3var_binomial()

    mktempdir() do tmp
        file = "$tmp/example_binomial.jpc"
        write(file, pc)

        pc2 = read(file, ProbCircuit)
        test_pc_equals(pc, pc2)

        pc2 = read(file, ProbCircuit, JpcFormat(), true)
        test_pc_equals(pc, pc2)

        pc2 = read(file, ProbCircuit, JpcFormat(), false)
        test_pc_equals(pc, pc2)

        # Compressed
        file = "$tmp/example_binomial.jpc.gz"
        write(file, pc)

        pc2 = read(file, ProbCircuit)
        test_pc_equals(pc, pc2)

    end
end

@testset "JPC IO tests Gaussian" begin
    pc = little_gmm()

    mktempdir() do tmp
        file = "$tmp/example_gaussian.jpc"
        write(file, pc)

        pc2 = read(file, ProbCircuit)
        test_pc_equals(pc, pc2)

        pc2 = read(file, ProbCircuit, JpcFormat(), true)
        test_pc_equals(pc, pc2)

        pc2 = read(file, ProbCircuit, JpcFormat(), false)
        test_pc_equals(pc, pc2)

        # Compressed
        file = "$tmp/example_gaussian.jpc.gz"
        write(file, pc)

        pc2 = read(file, ProbCircuit)
        test_pc_equals(pc, pc2)

    end
end

@testset "JPC IO tests 2D Gaussian" begin
    pc = little_2var_gmm()

    mktempdir() do tmp
        file = "$tmp/example_gaussian2.jpc"
        write(file, pc)

        pc2 = read(file, ProbCircuit)
        test_pc_equals(pc, pc2)

        pc2 = read(file, ProbCircuit, JpcFormat(), true)
        test_pc_equals(pc, pc2)

        pc2 = read(file, ProbCircuit, JpcFormat(), false)
        test_pc_equals(pc, pc2)

        # Compressed
        file = "$tmp/example_gaussian2.jpc.gz"
        write(file, pc)

        pc2 = read(file, ProbCircuit)
        test_pc_equals(pc, pc2)

    end
end


@testset "Jpc IO tests hybrid" begin
    
    pc = little_hybrid_circuit()

    mktempdir() do tmp
        
        file = "$tmp/example.jpc"
        write(file, pc)

        pc2 = read(file, ProbCircuit)
        test_pc_equals(pc, pc2)

        pc2 = read(file, ProbCircuit, JpcFormat(), true)
        test_pc_equals(pc, pc2)

        pc2 = read(file, ProbCircuit, JpcFormat(), false)
        test_pc_equals(pc, pc2)

        file = "$tmp/example.jpc.gz"
        write(file, pc)

        pc2 = read(file, ProbCircuit)
        test_pc_equals(pc, pc2)
        
    end

end
