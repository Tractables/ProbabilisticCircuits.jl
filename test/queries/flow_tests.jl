using Test, DirectedAcyclicGraphs, ProbabilisticCircuits
using ProbabilisticCircuits: PlainSumNode, PlainMulNode, PlainProbCircuit
using CUDA
import ProbabilisticCircuits as PCs

include("../helper/plain_dummy_circuits.jl")

@testset "flow" begin

    # LiteralDist
    
    pc = little_3var()
    bpc = bit_circuit(pc)

    data = cu([true true false; false true false; false false false])

    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)

    mars = PCs.prep_memory(nothing, (3, num_nodes), (false, true))
    flows = PCs.prep_memory(nothing, (3, num_nodes), (false, true))
    edge_aggr = PCs.prep_memory(nothing, (num_edges,))
    inparams_aggr = PCs.prep_memory(nothing, (length(bpc.input_node_idxs), bpc.inparams_aggr_size), (true, true))

    example_ids = cu(Int32.([1, 2, 3]))

    PCs.probs_flows_circuit(flows, mars, edge_aggr, inparams_aggr, bpc, data, example_ids; mine = 2, maxe = 32)
    edge_aggr_cpu = Array(edge_aggr)

    @test edge_aggr_cpu[2] ≈ Float32(3.0)
    @test edge_aggr_cpu[4] ≈ Float32(2.0)
    @test edge_aggr_cpu[6] ≈ Float32(1.0)

    # BernoulliDist

    pc = little_3var_bernoulli()
    bpc = bit_circuit(pc)

    data = cu([true true false; false true false; false false false])

    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)

    mars = PCs.prep_memory(nothing, (3, num_nodes), (false, true))
    flows = PCs.prep_memory(nothing, (3, num_nodes), (false, true))
    edge_aggr = PCs.prep_memory(nothing, (num_edges,))
    inparams_aggr = PCs.prep_memory(nothing, (length(bpc.input_node_idxs), bpc.inparams_aggr_size), (true, true))

    example_ids = cu(Int32.([1, 2, 3]))

    PCs.probs_flows_circuit(flows, mars, edge_aggr, inparams_aggr, bpc, data, example_ids; mine = 2, maxe = 32)
    inparams_aggr_cpu = Array(inparams_aggr)

    @test all(inparams_aggr_cpu .≈ Float32[1.0 2.0; 2.0 1.0; 0.0 3.0])

    # CategoricalDist

    pc = little_3var_categorical(; num_cats = UInt32(5))
    bpc = bit_circuit(pc)

    data = cu(UInt32.([2 3 4; 5 1 2; 3 4 5]))

    num_nodes = length(bpc.nodes)
    num_edges = length(bpc.edge_layers_down.vectors)

    mars = PCs.prep_memory(nothing, (3, num_nodes), (false, true))
    flows = PCs.prep_memory(nothing, (3, num_nodes), (false, true))
    edge_aggr = PCs.prep_memory(nothing, (num_edges,))
    inparams_aggr = PCs.prep_memory(nothing, (length(bpc.input_node_idxs), bpc.inparams_aggr_size), (true, true))

    example_ids = cu(Int32.([1, 2, 3]))

    PCs.probs_flows_circuit(flows, mars, edge_aggr, inparams_aggr, bpc, data, example_ids; mine = 2, maxe = 32)
    inparams_aggr_cpu = Array(inparams_aggr)

    @test all(inparams_aggr_cpu .≈ Float32[0.0 1.0 1.0 0.0 1.0; 1.0 0.0 1.0 1.0 0.0; 0.0 1.0 0.0 1.0 1.0])
    
end