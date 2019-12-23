using Test
using LogicCircuits
using ProbabilisticCircuits


@testset "pr_constraint Query" begin
    # two nodes
    simplevtree = zoo_vtree_file("simple2.vtree")
    pc, vtree = load_struct_prob_circuit(
                    zoo_psdd_file("simple2.4.psdd"), simplevtree)

    cache = Dict{Tuple{ProbΔNode, Union{ProbΔNode, StructLogicalΔNode}}, Float64}()

    @test abs(pr_constraint(pc[end], pc[end], cache) - 1.0) < 1e-8
    @test abs(pr_constraint(pc[5], pc[3], cache) - 0.2) < 1e-8
    @test abs(pr_constraint(pc[5], pc[4], cache) - 0.8) < 1e-8

    file_circuit = "little_4var.circuit"
    file_vtree = "little_4var.vtree"
    logical_circuit, vtree = load_struct_smooth_logical_circuit(
                                zoo_lc_file(file_circuit), zoo_vtree_file(file_vtree))

    pc = zoo_psdd("little_4var.psdd")

    @test abs(pr_constraint(pc[end], logical_circuit[end - 1], cache) - 1.0) < 1e-8

    # Test with two psdds
    pc1, vtree = load_struct_prob_circuit(zoo_psdd_file("simple2.5.psdd"), simplevtree)
    pc2, vtree = load_struct_prob_circuit(zoo_psdd_file("simple2.6.psdd"), simplevtree)

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