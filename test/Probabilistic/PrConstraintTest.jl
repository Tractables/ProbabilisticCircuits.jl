using Test
using .Juice


@testset "pr_constraint Query" begin
    # two nodes
    clt = parse_clt("./test/circuits/2.clt");
    vtree = learn_vtree_from_clt(clt; vtree_mode="balanced");
    (pc, bases) = compile_psdd_from_clt(clt, vtree);
    parents = parents_vector(pc);
    psdd = PSDDWrapper(pc, bases, parents, vtree)

    split_operation(psdd.pc[end], psdd.pc[7], Var.(1), psdd; depth = 0)
    split_operation(pc[end], pc[7], Var.(2), psdd; depth = 0)

    psdd.pc[9].log_thetas = [
        log(0.5),
        log(0.25),
        log(0.25)
    ]

    psdd.pc[5].log_thetas = [
        log(0.2),
        log(0.8)
    ]

    cache = Dict{Tuple{ProbΔNode, Union{ProbΔNode, StructLogicalΔNode}}, Float64}()

    @test abs(pr_constraint(psdd.pc[end], psdd.pc[end], cache) - 1.0) < 1e-8
    @test abs(pr_constraint(psdd.pc[5], psdd.pc[3], cache) - 0.2) < 1e-8
    @test abs(pr_constraint(psdd.pc[5], psdd.pc[4], cache) - 0.8) < 1e-8

    file_circuit = "test/circuits/little_4var.circuit"
    file_vtree = "test/circuits/little_4var.vtree"
    logical_circuit, vtree = load_struct_smooth_logical_circuit(file_circuit, file_vtree)

    file = "test/circuits/little_4var.psdd"
    pc = load_prob_circuit(file)

    @test abs(pr_constraint(pc[end], logical_circuit[end - 1], cache) - 1.0) < 1e-8

    # Test with two psdds
    clt1 = parse_clt("./test/circuits/2.clt");
    vtree1 = learn_vtree_from_clt(clt1, vtree_mode="balanced");
    (pc1, bases1) = compile_psdd_from_clt(clt1, vtree1);
    parents1 = parents_vector(pc1);
    psdd1 = PSDDWrapper(pc1, bases1, parents1, vtree1)

    split_operation(pc1[end], pc1[7], Var.(1), psdd1; depth = 0);
    split_operation(pc1[end], pc1[6], Var.(2), psdd1; depth = 0);

    pc1[9].log_thetas = map(log, [0.5, 0.25, 0.25]);
    pc1[5].log_thetas = map(log, [0.2, 0.8]);

    clt2 = parse_clt("./test/circuits/2.clt");
    vtree2 = learn_vtree_from_clt(clt2, vtree_mode="balanced");
    (pc2, bases2) = compile_psdd_from_clt(clt2, vtree2);
    parents2 = parents_vector(pc2)
    psdd2 = PSDDWrapper(pc2, bases2, parents2, vtree2)

    split_operation(pc2[end], pc2[7], Var.(2), psdd2; depth = 0)

    pc2[8].log_thetas = map(log, [0.3, 0.7])
    pc2[3].log_thetas = map(log, [0.9, 0.1])

    pr_constraint_cache = Dict{Tuple{ProbΔNode, Union{ProbΔNode, StructLogicalΔNode}}, Float64}()
    pr_constraint(pc1[end], pc2[end], pr_constraint_cache)
    @test abs(pr_constraint_cache[pc1[1], pc2[1]] - 1.0) < 1e-8
    @test abs(pr_constraint_cache[pc1[1], pc2[2]] - 0.0) < 1e-8
    @test abs(pr_constraint_cache[pc1[3], pc2[4]] - 1.0) < 1e-8
    @test abs(pr_constraint_cache[pc1[3], pc2[5]] - 0.0) < 1e-8
    @test abs(pr_constraint_cache[pc1[9], pc2[8]] - 1.0) < 1e-8
    @test abs(pr_constraint_cache[pc1[5], pc2[4]] - 0.2) < 1e-8
    @test abs(pr_constraint_cache[pc1[5], pc2[5]] - 0.8) < 1e-8
    @test abs(pr_constraint_cache[pc1[2], pc2[3]] - 1.0) < 1e-8
end