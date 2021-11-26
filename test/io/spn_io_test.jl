using Test
using ProbabilisticCircuits

include("../helper/pc_equals.jl")

@testset "Load and save a small SPN" begin
    
    function test_my_circuit(pc)
    
        @test pc isa ProbCircuit
    
        # Testing number of nodes and parameters
        @test  9 == num_parameters(pc)
        @test 20 == num_nodes(pc)
        
        # Testing Read Parameters
        EPS = 1e-7
        or1 = children(children(pc)[1])[2]
        @test abs(or1.log_probs[1] - (-1.6094379124341003)) < EPS
        @test abs(or1.log_probs[2] - (-1.2039728043259361)) < EPS
        @test abs(or1.log_probs[3] - (-0.916290731874155))  < EPS
        @test abs(or1.log_probs[4] - (-2.3025850929940455)) < EPS
    
        or2 = children(children(pc)[1])[1]
        @test abs(or2.log_probs[1] - (-2.3025850929940455))  < EPS
        @test abs(or2.log_probs[2] - (-2.3025850929940455))  < EPS
        @test abs(or2.log_probs[3] - (-2.3025850929940455))  < EPS
        @test abs(or2.log_probs[4] - (-0.35667494393873245)) < EPS
    
        @test abs(pc.log_probs[1] - (0.0)) < EPS
        @test check_parameter_integrity(pc)
    end

    # first load circuit from PSDD file
    pc1 = zoo_psdd("little_4var.psdd")

    mktempdir() do tmp
        
        spn_path = "$tmp/example.spn"
        write(spn_path, pc1)

        pc2 = read(spn_path, ProbCircuit)
        
        test_my_circuit(pc2)
        test_pc_equals(pc1, pc2)

        # try compressed
        write("$spn_path.gz", pc1)
        pc2 = read("$spn_path.gz", ProbCircuit)
        
        test_my_circuit(pc2)
        test_pc_equals(pc1, pc2)

    end

end
 
@testset "Can save SPNs with nonbinary multiplications" begin
    
    pc1 = fully_factorized_circuit(ProbCircuit, 10)
    
    mktempdir() do tmp
            
        spn_path = "$tmp/example.spn"
        write(spn_path, pc1)

        pc2 = read(spn_path, ProbCircuit)
        
        test_pc_equals(pc1, pc2)
    end
    
end