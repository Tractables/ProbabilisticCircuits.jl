using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames




@testset "Random Region Graph tests" begin

    num_vars = 10
    depth = 2
    replicas = 2
    
    region_graph = random_region_graph([Var(i) for i=1:num_vars]; depth, replicas);

    @test isnothing(region_graph.parent)
    @test typeof(region_graph) == RegionGraphInnerNode
    @test length(region_graph.paritions) == 2 # 2 replices means 2 partitions in root
    @test variables(region_graph) == BitSet([Var(i) for i=1:num_vars])

    @test length(region_graph.paritions[1]) == 5

    # Each parition should include the same set of variables
    prev_scope = nothing
    for cur_partition in region_graph.partitions
        cur_scope = variables(cur_partition[1])
        for i = 2:length(cur_partition)
            cur_scope = cur_scope ∪ variables(cur_partition[i])
        end
        
        if !isnothing(prev_scope)
            @test prev_scope == cur_scope "All paritions should include the same set of variables."
        else
            prev_scope = cur_scope
        end
    end


end

@testset "RAT SPN tests" begin

    data = DataFrame(BitArray([1 0 1 0 1 0 1 0 1 0;
        1 1 1 1 1 1 1 1 1 1;
        0 0 0 0 0 0 0 0 0 0;
        0 1 1 0 1 0 0 1 0 1]), :auto)

    num_vars = num_features(data)

    num_vars = 10
    depth = 2
    replicas = 2

    region_graph = random_region_graph([Var(i) for i=1:num_vars]; depth, replicas);
    circuit = region_graph_2_pc(region_graph; num_nodes_root = 1, num_nodes_region, num_nodes_leaf)[1];

    @test typeof(circuit) == ProbCircuit

end