using Test
using ProbabilisticCircuits
using LogicCircuits: is⋁gate, is⋀gate

include("../helper/pc_equals.jl")

@testset "Load and save a small PSDD with and without vtree" begin
    
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

    pc0 = zoo_psdd("little_4var.psdd")
    
    test_my_circuit(pc0)

    paths = (zoo_psdd_file("little_4var.psdd"), zoo_vtree_file("little_4var.vtree"))
    formats = (PsddFormat(), VtreeFormat())
    pc1 = read(paths, StructProbCircuit, formats) 

    @test pc1 isa StructProbCircuit
    test_my_circuit(pc1)
    @test respects_vtree(pc1, vtree(pc1))
    test_pc_equals(pc0, pc1)

    mktempdir() do tmp
        
        # write as a unstructured logic circuit
        psdd_path = "$tmp/example.psdd"
        write(psdd_path, pc1)

        # read as a unstructured logic circuit
        pc2 = read(psdd_path, ProbCircuit)
        
        test_my_circuit(pc2)
        test_pc_equals(pc0, pc2)

        # write with vtree
        vtree_path = "$tmp/example.vtree"
        paths = (psdd_path, vtree_path)
        write(paths, pc1)

        # read as a structured probabilistic circuit
        pc3 = read(paths, StructProbCircuit) 
        
        @test pc3 isa StructProbCircuit
        test_my_circuit(pc3)
        @test vtree(pc1) == vtree(pc3)
        test_pc_equals(pc0, pc3)

        # read/write compressed
        write("$psdd_path.gz", pc1)
        pc2 = read("$psdd_path.gz", ProbCircuit)
        
        test_my_circuit(pc2)
        test_pc_equals(pc1, pc2)

        # read/write compressed structured
        paths = ("$psdd_path.gz", vtree_path)
        write(paths, pc1)
        pc3 = read(paths, StructProbCircuit) 
        
        @test pc3 isa StructProbCircuit
        test_my_circuit(pc3)
        test_pc_equals(pc1, pc3)
        @test vtree(pc1) == vtree(pc3)
    end

end
 
 psdd_files = ["little_4var.psdd", "msnbc-yitao-a.psdd", "msnbc-yitao-b.psdd", "msnbc-yitao-c.psdd", "msnbc-yitao-d.psdd", "msnbc-yitao-e.psdd", "mnist-antonio.psdd"]
 
 @testset "Test parameter integrity of loaded PSDDs" begin
    for psdd_file in psdd_files
       @test check_parameter_integrity(zoo_psdd(psdd_file))
    end
 end

@testset "Cannot save PSDDs with nonbinary multiplications" begin
    
    pc = fully_factorized_circuit(ProbCircuit, 10)
    
    mktempdir() do tmp
            
        psdd_path = "$tmp/example.psdd"
        @test_throws ErrorException write(psdd_path, pc)

    end
end