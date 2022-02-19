using Test
using LogicCircuits
using ProbabilisticCircuits

@testset "Probability of constraint" begin

    # two nodes
    simplevtree = zoo_vtree_file("simple2.vtree")
    simplepsdd = zoo_psdd_file("simple2.4.psdd")
    pc = read((simplepsdd, simplevtree), StructProbCircuit)
    
    @test pr_constraint(pc, pc) ≈ 1.0

    # Test with two psdds
    pc1files = (zoo_psdd_file("simple2.5.psdd"), simplevtree)
    pc2files = (zoo_psdd_file("simple2.6.psdd"), simplevtree)
    pc1 = read(pc1files, StructProbCircuit)
    pc2 = read(pc2files, StructProbCircuit)

    @test pr_constraint(pc1, pc2) ≈ 1

end
