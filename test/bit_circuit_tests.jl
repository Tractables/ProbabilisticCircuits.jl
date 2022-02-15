using Test, ProbabilisticCircuits, CUDA
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, 
    update_parameters, SumEdge, BitsInput
using DirectedAcyclicGraphs: left_most_descendent

include("helper/plain_dummy_circuits.jl")

@testset "BitsPC tests" begin
    
    # Indicators
    pc = little_3var()
    bpc = BitsProbCircuit(pc)
    @test bpc isa BitsProbCircuit
    @test length(bpc.input_node_ids) == 6
    @test length(bpc.nodes) == 11
    @test length(bpc.heap) == 0

    x,y = bpc.edge_layers_up.vectors[1:2]
    newparams = log.([0.99, 0.01])
    bpc.edge_layers_up.vectors[1] = 
        SumEdge(x.parent_id, x.prime_id, x.sub_id, newparams[1], x.tag)
    bpc.edge_layers_up.vectors[2] = 
        SumEdge(y.parent_id, y.prime_id, y.sub_id, newparams[2], y.tag)
    update_parameters(bpc)
    @test all(params(sumnodes(pc)[1]) .≈ newparams)
    
    CUDA.@allowscalar if CUDA.functional()
        cbpc = cu(bpc)
        @test length(cbpc.input_node_ids) == 6
        @test length(cbpc.nodes) == 11
        @test length(cbpc.heap) == 0

        x,y = cbpc.edge_layers_up.vectors[1:2]
        newparams = log.([0.29, 0.71])
        cbpc.edge_layers_up.vectors[1] = 
            SumEdge(x.parent_id, x.prime_id, x.sub_id, newparams[1], x.tag)
        cbpc.edge_layers_up.vectors[2] = 
            SumEdge(y.parent_id, y.prime_id, y.sub_id, newparams[2], y.tag)
        update_parameters(cbpc)
        @test all(params(sumnodes(pc)[1]) .≈ newparams)
    end

    # Bernoullis
    pc = little_3var_bernoulli()
    bpc = BitsProbCircuit(pc)
    @test bpc isa BitsProbCircuit
    @test length(bpc.input_node_ids) == 3
    @test length(bpc.nodes) == 5
    @test length(bpc.heap) == 12

    bpc.heap[dist(bpc.nodes[1]).heap_start] = log(0.12)
    update_parameters(bpc)
    @test dist(left_most_descendent(pc)).logp ≈ log(0.12)

    CUDA.@allowscalar if CUDA.functional()
        cbpc = cu(bpc)
        @test length(cbpc.input_node_ids) == 3
        @test length(cbpc.nodes) == 5
        @test length(cbpc.heap) == 12

        cbpc.heap[dist(cbpc.nodes[1]).heap_start] = log(0.22)
        update_parameters(cbpc)
        @test dist(left_most_descendent(pc)).logp ≈ log(0.22)
    end
    
    # Categoricals
    pc = little_3var_categorical(; num_cats = 5)
    bpc = BitsProbCircuit(pc)
    @test bpc isa BitsProbCircuit
    @test length(bpc.input_node_ids) == 3
    @test length(bpc.nodes) == 5
    @test length(bpc.heap) == (5*2+1)*3

    newparams = log.([0.1,0.1,0.1,0.3,0.4])
    bpc.heap[1:5] .=  newparams
    update_parameters(bpc)
    @test all(dist(left_most_descendent(pc)).logps .≈ newparams)

    CUDA.@allowscalar if CUDA.functional()
        cbpc = cu(bpc)
        @test length(cbpc.input_node_ids) == 3
        @test length(cbpc.nodes) == 5
        @test length(cbpc.heap) == (5*2+1)*3

        newparams = log.([0.1,0.1,0.3,0.4,0.1])
        cbpc.heap[1:5] .=  CuVector(newparams)
        update_parameters(cbpc)
        @test all(dist(left_most_descendent(pc)).logps .≈ newparams)
    end
end
