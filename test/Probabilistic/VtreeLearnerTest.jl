using Test
using LogicCircuits
using ProbabilisticCircuits

# function vtree_test_top_down()
#     vars = Var.([1,2,3,4,5,6])
#     vtree = top_down_vtree(vars, test_top_down)
#     save(vtree, "circuits/vtree/vtree-test-top-down.vtree.dot")
#     return vtree
# end

# function vtree_test_bottom_up()
#     vars = Var.([1,2,3,4,5,6])
#     vtree = bottom_up_vtree(vars, test_bottom_up!)
#     save(vtree, "circuits/vtree/vtree-test-bottom-up.vtree.dot")
#     return vtree
# end

# function vtree_blossom_simply()
#     # even
#     vars = Var.([1,2,3,4])
#     mi = [  0.0 3.0 9.0 6.0;
#             3.0 0.0 5.0 8.0;
#             9.0 5.0 0.0 7.0;
#             6.0 8.0 7.0 0.0]
#     context = BlossomContext(vars, mi)
#     vtree = bottom_up_vtree(vars, blossom_bottom_up_curry(context))
#     save(vtree, "circuits/vtree/vtree-blossom-bottom-up-even.vtree.dot")

#     # odd
#     vars = Var.([1, 2, 3, 4, 5])
#     mi = [  0.0 3.0 9.0 6.0 1.0;
#             3.0 0.0 5.0 8.0 4.0;
#             9.0 5.0 0.0 7.0 3.0;
#             6.0 8.0 7.0 0.0 2.0;
#             1.0 4.0 3.0 2.0 0.0]
#     context = BlossomContext(vars, mi)
#     vtree = bottom_up_vtree(vars, blossom_bottom_up_curry(context))
#     save(vtree, "circuits/vtree/vtree-blossom-bottom-up-odd.vtree.dot")
# end

# function check_equality()
#     for name in twenty_dataset_names
#         for method in ["miMetis", "miBlossom"]
#             scala_vtree_path = "./report/resources/scala-vtree/$method/$name/$name.vtree";
#             julia_vtree_path = "./report/resources/julia-vtree/$method/$method-$name.vtree";
#             scala_vtree = load_vtree(scala_vtree_path);
#             julia_vtree = load_vtree(julia_vtree_path);
#             save(scala_vtree, scala_vtree_path*".dot")
#             if isequal_unordered(scala_vtree, julia_vtree)
#                 println("$method, $name")
#             end
#         end
#     end
# end

@testset "PSDD and PlainVtree Learner Test" begin
    data = dataset(twenty_datasets("nltcs"); do_shuffle=false, batch_size=-1);
    train_data = train(data);
    clt = learn_chow_liu_tree(WXData(train_data));
    #clt = parse_clt("circuits/test.clt")
    vtree = learn_vtree_from_clt(clt; vtree_mode="balanced"); # or "linear"
    @test vtree isa PlainVtree

    mktempdir() do tmp
        save(vtree, "$tmp/test.vtree.dot")        
        (psdd, _) = compile_psdd_from_clt(clt, vtree);
        @test psdd isa ProbΔ
        save_as_dot(psdd, "$tmp/test.psdd.dot")
    end
    
end
