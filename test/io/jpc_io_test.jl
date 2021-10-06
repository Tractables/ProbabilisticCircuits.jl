using Test
using ProbabilisticCircuits

include("../helper/pc_equals.jl")

@testset "Load and save a small JPC" begin
    
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
    paths = (zoo_psdd_file("little_4var.psdd"), zoo_vtree_file("little_4var.vtree"))
    formats = (PsddFormat(), VtreeFormat())
    pc1 = read(paths, StructProbCircuit, formats) 

    mktempdir() do tmp
        
        # write as a unstructured logic circuit
        jpc_path = "$tmp/example.jpc"
        write(jpc_path, pc1)

        # read as a unstructured logic circuit
        pc2 = read(jpc_path, ProbCircuit)
        
        test_my_circuit(pc2)
        test_pc_equals(pc1, pc2)

        # write with vtree
        vtree_path = "$tmp/example.vtree"
        paths = (jpc_path, vtree_path)
        write(paths, pc1)

        # read as a structured probabilistic circuit
        pc3 = read(paths, StructProbCircuit) 
        
        @test pc3 isa StructProbCircuit
        test_my_circuit(pc3)
        test_pc_equals(pc1, pc3)
        @test vtree(pc1) == vtree(pc3)
        
    end

end
 
@testset "Can save JPCs with nonbinary multiplications" begin
    
    pc1 = fully_factorized_circuit(ProbCircuit, 10)
    
    mktempdir() do tmp
            
        jpc_path = "$tmp/example.jpc"
        write(jpc_path, pc1)

        pc2 = read(jpc_path, ProbCircuit)
        
        test_pc_equals(pc1, pc2)
    end
    
end