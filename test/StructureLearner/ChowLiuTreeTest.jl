using Test: @test, @testset
using LightGraphs: nv, ne
using MetaGraphs: get_prop
using LogicCircuits
using ProbabilisticCircuits

@testset "Chow-Liu Tree learner tests" begin
    data = dataset(twenty_datasets("nltcs"); do_shuffle=false, batch_size=-1)
    train_x = train(data)
    clt = learn_chow_liu_tree(train_x; α=1.0, clt_root="graph_center")
    pv = parent_vector(clt)

    @test clt isa CLT
    @test nv(clt) == num_features(train_x)
    @test ne(clt) == num_features(train_x) - 1 

    root = findall(x -> x == 0, pv)[1]
    cpt = get_prop(clt, root, :cpt)
    @test cpt[0] ≈ 0.7828236021007106
    @test cpt[0] + cpt[1] ≈ 1 
    
    cpt1 = get_prop(clt, 1, :cpt)
    @test cpt1[(0,0)] + cpt1[(1, 0)] ≈ 1
    @test cpt1[(0,1)] + cpt1[(1, 1)] ≈ 1
    @test get_prop(clt, 1, :parent) == pv[1]

end