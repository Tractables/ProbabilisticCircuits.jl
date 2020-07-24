using Test
using LogicCircuits
using ProbabilisticCircuits

@testset "PSDD and PlainVtree Learner Test" begin
    data = dataset(twenty_datasets("nltcs"); do_shuffle=false, batch_size=-1);
    train_data = train(data);
    clt = learn_chow_liu_tree(WXData(train_data));
    #clt = parse_clt("circuits/test.clt")
    vtree = learn_vtree_from_clt(clt; vtree_mode="balanced"); # or "linear"
    @test vtree isa PlainVtree

    mktempdir() do tmp
        save(vtree, "$tmp/test.vtree.dot")        
        psdd = compile_psdd_from_clt(clt, vtree);
        @test psdd isa ProbΔ
        save_as_dot(psdd, "$tmp/test.psdd.dot")
    end
    
end
