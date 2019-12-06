using Test
using .Juice

@testset "Entropy and KLD" begin
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

    clt3 = parse_clt("./test/circuits/2.clt");
    vtree3 = learn_vtree_from_clt(clt2, vtree_mode="balanced");
    (pc3, bases3) = compile_psdd_from_clt(clt3, vtree3);
    parents3 = parents_vector(pc3)
    psdd3 = PSDDWrapper(pc3, bases3, parents3, vtree3)

    split_operation(pc3[end], pc3[7], Var.(2), psdd3; depth = 0)

    pc3[8].log_thetas = map(log, [0.4, 0.6])
    pc3[3].log_thetas = map(log, [0.5, 0.5])

    # Entropy calculation test
    @test abs(psdd_entropy(pc1[end]) - 1.2899219826090118) < 1e-8
    @test abs(psdd_entropy(pc2[end]) - 0.9359472745536583) < 1e-8

    # KLD calculation test
    pr_constraint_cache = Dict{Tuple{ProbΔNode, Union{ProbΔNode, StructLogicalΔNode}}, Float64}()

    kl_divergence_cache = Dict{Tuple{ProbΔNode, ProbΔNode}, Float64}()
    @test abs(psdd_kl_divergence(pc1[1], pc2[1], kl_divergence_cache, pr_constraint_cache) - 0.0) < 1e-8
    @test abs(psdd_kl_divergence(pc1[1], pc2[3], kl_divergence_cache, pr_constraint_cache) + log(0.9)) < 1e-8
    @test abs(psdd_kl_divergence(pc1[2], pc2[3], kl_divergence_cache, pr_constraint_cache) + log(0.1)) < 1e-8
    @test abs(psdd_kl_divergence(pc1[5], pc2[4], kl_divergence_cache, pr_constraint_cache) - 0.2 * log(0.2)) < 1e-8
    @test abs(psdd_kl_divergence(pc1[5], pc2[5], kl_divergence_cache, pr_constraint_cache) - 0.8 * log(0.8)) < 1e-8
    @test abs(psdd_kl_divergence(pc1[5], pc2[5], kl_divergence_cache, pr_constraint_cache) - 0.8 * log(0.8)) < 1e-8
    @test abs(psdd_kl_divergence(pc1[end], pc2[end]) - 0.5672800167911778) < 1e-8

    kl_divergence_cache = Dict{Tuple{ProbΔNode, ProbΔNode}, Float64}()
    @test abs(psdd_kl_divergence(pc2[4], pc3[5], kl_divergence_cache, pr_constraint_cache) - 0.0) < 1e-8
    @test abs(psdd_kl_divergence(pc2[4], pc3[4], kl_divergence_cache, pr_constraint_cache) - 0.0) < 1e-8
    @test abs(psdd_kl_divergence(pc2[3], pc3[3], kl_divergence_cache, pr_constraint_cache) - 0.9 * log(0.9 / 0.5) - 0.1 * log(0.1 / 0.5)) < 1e-8
    @test abs(psdd_kl_divergence(pc2[end], pc3[end]) - 0.38966506) < 1e-8

end
