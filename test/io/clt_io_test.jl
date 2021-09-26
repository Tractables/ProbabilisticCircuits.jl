using Test
using ProbabilisticCircuits
using Suppressor
using LightGraphs: nv, ne
using MetaGraphs: get_prop

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
