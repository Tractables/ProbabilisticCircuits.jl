using Test
using LogicCircuits
using ProbabilisticCircuits

@testset "Probability of constraint" begin

    # two nodes
    simplevtree = zoo_vtree_file("simple2.vtree")
    pc, vtree = load_struct_prob_circuit(
                    zoo_psdd_file("simple2.4.psdd"), simplevtree)

  
    @test pr_constraint(pc, pc) ≈ 1.0

    file_circuit = "little_4var.circuit"
    file_vtree = "little_4var.vtree"
    logic_circuit, vtree = load_struct_smooth_logic_circuit(
                                zoo_lc_file(file_circuit), zoo_vtree_file(file_vtree))

    pc, _ = load_struct_prob_circuit(zoo_psdd_file("little_4var.psdd"), zoo_vtree_file("little_4var.vtree"))

    @test pr_constraint(pc, children(logic_circuit)[1]) ≈ 1.0

    # Test with two psdds
    pc1, vtree = load_struct_prob_circuit(zoo_psdd_file("simple2.5.psdd"), simplevtree)
    pc2, vtree = load_struct_prob_circuit(zoo_psdd_file("simple2.6.psdd"), simplevtree)

    @test pr_constraint(pc1, pc2) ≈ 1

end
