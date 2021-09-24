using Test
using ProbabilisticCircuits

@testset "Load a small PSDD and test methods" begin
    prob_circuit = zoo_psdd("little_4var.psdd")
    @test prob_circuit isa ProbCircuit
 
    # Testing number of nodes and parameters
    @test  9 == num_parameters(prob_circuit)
    @test 20 == num_nodes(prob_circuit)
    
    # Testing Read Parameters
    EPS = 1e-7
    or1 = children(children(prob_circuit)[1])[2]
    @test abs(or1.log_probs[1] - (-1.6094379124341003)) < EPS
    @test abs(or1.log_probs[2] - (-1.2039728043259361)) < EPS
    @test abs(or1.log_probs[3] - (-0.916290731874155)) < EPS
    @test abs(or1.log_probs[4] - (-2.3025850929940455)) < EPS
 
    or2 = children(children(prob_circuit)[1])[1]
    @test abs(or2.log_probs[1] - (-2.3025850929940455)) < EPS
    @test abs(or2.log_probs[2] - (-2.3025850929940455)) < EPS
    @test abs(or2.log_probs[3] - (-2.3025850929940455)) < EPS
    @test abs(or2.log_probs[4] - (-0.35667494393873245)) < EPS
 
    @test abs(prob_circuit.log_probs[1] - (0.0)) < EPS
 end
 
 psdd_files = ["little_4var.psdd", "msnbc-yitao-a.psdd", "msnbc-yitao-b.psdd", "msnbc-yitao-c.psdd", "msnbc-yitao-d.psdd", "msnbc-yitao-e.psdd", "mnist-antonio.psdd"]
 
 @testset "Test parameter integrity of loaded PSDDs" begin
    for psdd_file in psdd_files
       @test check_parameter_integrity(zoo_psdd(psdd_file))
    end
 end