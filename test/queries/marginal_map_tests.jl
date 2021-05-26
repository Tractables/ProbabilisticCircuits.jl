using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame
using StatsBase

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

@testset "conditioning" begin
    prob_circ = zoo_psdd("asia.uai.psdd")
    pc = pc_condition(prob_circ, Var(3), Var(4), Var(7))
    pc_cond = pc_condition(pc, Var(1), Var(8))
    @test get_margs(pc_cond, 8, [2,5], []) ≈ get_margs(pc, 8, [2,5], [1,8]) atol = 1e-6
end


@testset "splitting" begin
    prob_circ = zoo_psdd("asia.uai.psdd")
    
    for i in 1:5
        @show quer = BitSet(StatsBase.sample(1:8, 4, replace=false))
        qc = map(x -> Var(x), collect(quer))
        mmap = brute_force_mmap(prob_circ, quer)
        split_circ = add_and_split(prob_circ, qc[1])
        split_circ2 = add_and_split(split_circ, qc[2])
        new_mmap = brute_force_mmap(split_circ2, quer)
        # @test forward_bounds(split_circ2, quer)[split_circ2] ≈ mmap atol=1e-6
        @test mmap ≈ new_mmap atol=1e-5
        @test get_margs(prob_circ, 8, [qc[3], qc[4]], []) ≈ get_margs(split_circ, 8, [qc[3], qc[4]], []) atol=1e-5
    end
end