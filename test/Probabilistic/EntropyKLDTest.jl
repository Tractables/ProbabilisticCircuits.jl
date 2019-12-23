using Test
using LogicCircuits
using ProbabilisticCircuits

@testset "Entropy and KLD" begin
    pc1, vtree = load_struct_prob_circuit(
                    zoo_psdd_file("simple2.1.psdd"), zoo_vtree_file("simple2.vtree"))
    pc2, vtree = load_struct_prob_circuit(
                    zoo_psdd_file("simple2.2.psdd"), zoo_vtree_file("simple2.vtree"))
    pc3, vtree = load_struct_prob_circuit(
                    zoo_psdd_file("simple2.3.psdd"), zoo_vtree_file("simple2.vtree"))
   
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
