using Test
using LogicCircuits
using ProbabilisticCircuits
using LightGraphs
using MetaGraphs
using Suppressor



@testset "Test parameter integrity of loaded structured PSDDs" begin
   circuit, vtree = load_struct_prob_circuit(
      zoo_psdd_file("little_4var.psdd"), zoo_vtree_file("little_4var.vtree"))
   @test check_parameter_integrity(circuit)
   @test vtree isa PlainVtree
   @test respects_vtree(circuit, vtree)
end

@testset "Test loaded logistic circuits" begin
   classes = 2
   lc = zoo_lc("little_4var.circuit", classes)
   @test lc isa LogisticCircuit
   @test num_nodes(lc) == 29
   @test num_parameters(lc) == num_parameters_per_class(lc) * classes
   @test check_parameter_integrity(lc)
   @test all(lc.thetas ≈ [0.6803363307333976 0.7979990493834855])
   @test all(lc.children[1].thetas ≈ [0.34978385969447856 0.8620937209951014])
   or1 = lc.children[1].children[1].children[1]
   @test all(or1.thetas ≈ [0.012723368087276588 0.44600859247854274;
                           0.4126851950485019 0.8204476705654302;
                           0.2633016148946008 0.010045227037839832;
                           0.27132714432580674 0.9677387544772587])
end


@testset "Test loaded clt file" begin
   # TODO add more clt files
   t = zoo_clt("4.clt")
   @test t isa CLT
   @test_nowarn @suppress_out print_tree(t)
   @test nv(t) == 4
   @test ne(t) == 0
   for (v, p) in enumerate([.2, .1, .3, .4])
      cpt = get_prop(t, v, :cpt)
      @test cpt[1] + cpt[0] ≈ 1.0
      @test cpt[1] ≈ p
   end
end
