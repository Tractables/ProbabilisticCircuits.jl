# This tests are suppoed to test queries on the circuits
using Test;

include("../src/Circuits.jl")

@testset "Probability of Full Evidence" begin

    prob_circuit = load_psdd_prob_circuit("test/circuits/little_4var.psdd");
    @test prob_circuit isa Vector{<:ProbCircuitNode};

    agg_circuit = AggregateFlowCircuit(prob_circuit, Int8)
    @test agg_circuit isa Vector{<:AggregateFlowCircuitNode};

    flow_circuit = FlowCircuit(agg_circuit, 1, Int8)
    @test flow_circuit isa Vector{<:FlowCircuitNode};

    # [0 0 0 0] -> 0.700
    # [0 1 1 0] -> 0.300
    # [0 0 1 1] -> 0.139
    
    println(pass_up(flow_circuit, [[0,0,0,1]]))

    ##@test 1 == 2; #incomplete test, make it fail for now

end