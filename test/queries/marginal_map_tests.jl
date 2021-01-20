using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame

@testset "Forward bound" begin
    prob_circuit = zoo_psdd("asia-bdd.psdd")

    data_marg = DataFrame([missing missing missing missing missing missing missing missing])
    map, mappr = MAP(prob_circuit, data_marg)
    fwd_mpe = forward_bounds(prob_circuit, BitSet([1,2,3,4,5,6,7,8]))[prob_circuit]
    @test mappr[1] ≈ fwd_mpe atol=1e-6

    wmc = MAR(prob_circuit, data_marg)[1]
    fwd_wmc = forward_bounds(prob_circuit, BitSet([]))[prob_circuit]
    @test wmc ≈ fwd_wmc atol=1e-6

    fwd_mmap = forward_bounds(prob_circuit, BitSet([1,2,3]))[prob_circuit]
    # Computed using merlin. The psdd is a bdd with the right variable order, so we'll compute it exactly
    @test fwd_mmap ≈ -1.059872 atol=1e-6
end