function vtree_test_top_down()
    vars = Set(Var.([1,2,3,4,5,6]))
    context = TestContext()
    vtree = construct_top_down(vars, test_top_down, context)
    save(vtree, "./test/circuits/vtree/vtree-test-top-down.vtree.dot")
    return vtree
end

function vtree_test_bottom_up()
    vars = Set(Var.([1,2,3,4,5,6]))
    context = TestContext()
    vtree = construct_bottom_up(vars, test_bottom_up!, context)
    save(vtree, "./test/circuits/vtree/vtree-test-bottom-up.vtree.dot")
    return vtree
end

function vtree_blossom_simply()
    # even
    vars = Set(Var.([1,2,3,4]))
    mi = [  0.0 3.0 9.0 6.0;
            3.0 0.0 5.0 8.0;
            9.0 5.0 0.0 7.0;
            6.0 8.0 7.0 0.0]
    context = BlossomContext(vars, mi)
    vtree = construct_bottom_up(vars, blossom_bottom_up!, context)
    save(vtree, "./test/circuits/vtree/vtree-blossom-bottom-up-even.vtree.dot")

    # odd
    vars = Set(Var.([1, 2, 3, 4, 5]))
    mi = [  0.0 3.0 9.0 6.0 1.0;
            3.0 0.0 5.0 8.0 4.0;
            9.0 5.0 0.0 7.0 3.0;
            6.0 8.0 7.0 0.0 2.0;
            1.0 4.0 3.0 2.0 0.0]
    context = BlossomContext(vars, mi)
    vtree = construct_bottom_up(vars, blossom_bottom_up!, context)
    save(vtree, "./test/circuits/vtree/vtree-blossom-bottom-up-odd.vtree.dot")
end

function check_equality()
    for name in twenty_dataset_names
        for method in ["miMetis", "miBlossom"]
            scala_vtree_path = "./report/resources/scala-vtree/$method/$name/$name.vtree";
            julia_vtree_path = "./report/resources/julia-vtree/$method/$method-$name.vtree";
            scala_vtree = load_vtree(scala_vtree_path);
            julia_vtree = load_vtree(julia_vtree_path);
            save(scala_vtree, scala_vtree_path*".dot")
            if isequal_unordered(scala_vtree, julia_vtree)
                println("$method, $name")
            end
        end
    end
end
