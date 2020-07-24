using Test
using LogicCircuits
using ProbabilisticCircuits

# TODO reinstate after fix tests by replacing indexing circuit node

@testset "Entropy and KLD" begin
    pc1, vtree = load_struct_prob_circuit(
                    zoo_psdd_file("simple2.1.psdd"), zoo_vtree_file("simple2.vtree"))
    pc2, vtree = load_struct_prob_circuit(
                    zoo_psdd_file("simple2.2.psdd"), zoo_vtree_file("simple2.vtree"))
    pc3, vtree = load_struct_prob_circuit(
                    zoo_psdd_file("simple2.3.psdd"), zoo_vtree_file("simple2.vtree"))
   
    # Entropy calculation test
    @test abs(entropy(pc1) - 1.2899219826090118) < 1e-8
    @test abs(entropy(pc2) - 0.9359472745536583) < 1e-8

    # KLD Tests #
    # KLD base tests
    pr_constraint_cache = Dict{Tuple{ProbCircuit, Union{ProbCircuit, StructLogicCircuit}}, Float64}()
    kl_divergence_cache = Dict{Tuple{ProbCircuit, ProbCircuit}, Float64}()

    # @test_throws AssertionError("Both nodes not normalized for same vtree node") kl_divergence(pc1[1], pc1[3], kl_divergence_cache, pr_constraint_cache)
    # @test_throws AssertionError("Both nodes not normalized for same vtree node") kl_divergence(pc1[2], pc1[3], kl_divergence_cache, pr_constraint_cache)
    # @test_throws AssertionError("Both nodes not normalized for same vtree node") kl_divergence(pc1[1], pc1[4], kl_divergence_cache, pr_constraint_cache)
    # @test_throws AssertionError("Both nodes not normalized for same vtree node") kl_divergence(pc1[1], pc1[5], kl_divergence_cache, pr_constraint_cache)
    # @test_throws AssertionError("Both nodes not normalized for same vtree node") kl_divergence(pc1[2], pc1[5], kl_divergence_cache, pr_constraint_cache)

    # @test_throws AssertionError("Prob⋀ not a valid PSDD node for KL-Divergence") kl_divergence(pc1[1], pc1[6], kl_divergence_cache, pr_constraint_cache)
    # @test_throws AssertionError("Prob⋀ not a valid PSDD node for KL-Divergence") kl_divergence(pc1[7], pc1[2], kl_divergence_cache, pr_constraint_cache)
    # @test_throws AssertionError("Prob⋀ not a valid PSDD node for KL-Divergence") kl_divergence(pc1[6], pc2[7], kl_divergence_cache, pr_constraint_cache)

    # KLD calculation test
    # @test abs(kl_divergence(pc1[1], pc2[1], kl_divergence_cache, pr_constraint_cache) - 0.0) < 1e-8
    # @test abs(kl_divergence(pc1[1], pc1[2], kl_divergence_cache, pr_constraint_cache) - 0.0) < 1e-8
    # @test abs(kl_divergence(pc1[1], pc2[3], kl_divergence_cache, pr_constraint_cache) + log(0.9)) < 1e-8
    # @test abs(kl_divergence(pc1[2], pc2[3], kl_divergence_cache, pr_constraint_cache) + log(0.1)) < 1e-8
    # @test abs(kl_divergence(pc1[5], pc2[4], kl_divergence_cache, pr_constraint_cache) - 0.2 * log(0.2)) < 1e-8
    # @test abs(kl_divergence(pc1[5], pc2[5], kl_divergence_cache, pr_constraint_cache) - 0.8 * log(0.8)) < 1e-8
    # @test abs(kl_divergence(pc1[5], pc2[5], kl_divergence_cache, pr_constraint_cache) - 0.8 * log(0.8)) < 1e-8
    @test abs(kl_divergence(pc1, pc2) - 0.5672800167911778) < 1e-8

    kl_divergence_cache = Dict{Tuple{ProbCircuit, ProbCircuit}, Float64}()
    # @test abs(kl_divergence(pc2[4], pc3[5], kl_divergence_cache, pr_constraint_cache) - 0.0) < 1e-8
    # @test abs(kl_divergence(pc2[4], pc3[4], kl_divergence_cache, pr_constraint_cache) - 0.0) < 1e-8
    # @test abs(kl_divergence(pc2[3], pc3[3], kl_divergence_cache, pr_constraint_cache) - 0.9 * log(0.9 / 0.5) - 0.1 * log(0.1 / 0.5)) < 1e-8
    @test abs(kl_divergence(pc2, pc3) - 0.38966506) < 1e-8

end

@testset "Pr constraint Query" begin
    # two nodes
    simplevtree = zoo_vtree_file("simple2.vtree")
    pc, vtree = load_struct_prob_circuit(
                    zoo_psdd_file("simple2.4.psdd"), simplevtree)

    cache = Dict{Tuple{ProbCircuit, Union{ProbCircuit, StructLogicCircuit}}, Float64}()

    @test abs(pr_constraint(pc, pc, cache) - 1.0) < 1e-8
    # @test abs(pr_constraint(pc[5], pc[3], cache) - 0.2) < 1e-8
    # @test abs(pr_constraint(pc[5], pc[4], cache) - 0.8) < 1e-8

    file_circuit = "little_4var.circuit"
    file_vtree = "little_4var.vtree"
    logic_circuit, vtree = load_struct_smooth_logic_circuit(
                                zoo_lc_file(file_circuit), zoo_vtree_file(file_vtree))

    pc = zoo_psdd("little_4var.psdd")

    @test abs(pr_constraint(pc, children(logic_circuit)[1], cache) - 1.0) < 1e-8

    # Test with two psdds
    pc1, vtree = load_struct_prob_circuit(zoo_psdd_file("simple2.5.psdd"), simplevtree)
    pc2, vtree = load_struct_prob_circuit(zoo_psdd_file("simple2.6.psdd"), simplevtree)

    pr_constraint_cache = Dict{Tuple{ProbCircuit, Union{ProbCircuit, StructLogicCircuit}}, Float64}()
    pr_constraint(pc1, pc2, pr_constraint_cache)
    # @test abs(pr_constraint_cache[pc1[1], pc2[1]] - 1.0) < 1e-8
    # @test abs(pr_constraint_cache[pc1[1], pc2[2]] - 0.0) < 1e-8
    # @test abs(pr_constraint_cache[pc1[3], pc2[4]] - 1.0) < 1e-8
    # @test abs(pr_constraint_cache[pc1[3], pc2[5]] - 0.0) < 1e-8
    # @test abs(pr_constraint_cache[pc1[9], pc2[8]] - 1.0) < 1e-8
    # @test abs(pr_constraint_cache[pc1[5], pc2[4]] - 0.2) < 1e-8
    # @test abs(pr_constraint_cache[pc1[5], pc2[5]] - 0.8) < 1e-8
    # @test abs(pr_constraint_cache[pc1[2], pc2[3]] - 1.0) < 1e-8
end
