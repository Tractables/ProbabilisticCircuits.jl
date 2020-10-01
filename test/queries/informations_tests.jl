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
   
    @test entropy(pc1) ≈ 1.2899219826090118
    @test entropy(pc2) ≈ 0.9359472745536583

    @test kl_divergence(pc1, pc2) ≈ 0.5672800167911778
    @test kl_divergence(pc2, pc3) ≈ 0.38966506

end
