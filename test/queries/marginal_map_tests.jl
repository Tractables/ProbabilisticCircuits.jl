using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame
using DataStructures: counter
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
        data_marg = DataFrame(repeat([missing], 1, num_variables(prob_circ)))
        _, mpe = MAP(prob_circ, data_marg)

        split_circ = add_and_split(prob_circ, qc[1])
        split_circ2 = add_and_split(split_circ, qc[2])
        new_mmap = brute_force_mmap(split_circ2, quer)
        data_marg = DataFrame(repeat([missing], 1, num_variables(split_circ2)))
        _, new_mpe = MAP(prob_circ, data_marg)

        # @test forward_bounds(split_circ2, quer)[split_circ2] ≈ mmap atol=1e-6
        @test mmap ≈ new_mmap atol=1e-5
        @test mpe ≈ new_mpe atol=1e-5
        @test get_margs(prob_circ, 8, [qc[3], qc[4]], []) ≈ get_margs(split_circ, 8, [qc[3], qc[4]], []) atol=1e-5
    end
end

@testset "Split heuristic" begin
    prob_circ = zoo_psdd("asia.uai.psdd")
    quer = BitSet([3,4,6])

    c = counter(Dict(-3 => 20, 6 => 5, -6 => 4))
    @test get_to_split(prob_circ, quer, c, "minD") == 6
    @test get_to_split(prob_circ, quer, c, "maxP") == 3
    @test get_to_split(prob_circ, quer, c, "maxDepth") == 4
end

@testset "maxsum" begin
    a,b,c = pos_literals(ProbCircuit, 3)
    circuit_1 = 0.5 * (a * b * (0.3 * c + 0.7 * -c)) + 0.5 * (a * -b * (0.8 * c + 0.2 * -c))
    circuit_2 = -a * (0.9 * (b * c) + 0.1 * (-b * -c))
    circuit = 0.6 * circuit_1 + 0.4 * circuit_2

    # make sure the assignment to query variable is retrieved even when it appears under a non-max node (circuit_1)
    # otherwise, the state for variable a would be left as the default value of false
    quer_a = BitSet([variable(a)])
    state, pr = max_sum_lower_bound(circuit, quer_a)
    @test state[1,variable(a)] == true 
    @test pr ≈ 0.6
    @test log(pr) ≈ MAR(circuit,state)[1]

    quer_ab = BitSet([variable(a),variable(b)])
    state, pr = max_sum_lower_bound(circuit, quer_ab)
    @test state[1,variable(a)] == false && state[1,variable(b)] == true
    @test pr ≈ 0.4 * 0.9
    @test log(pr) ≈ MAR(circuit,state)[1]

    quer_ac = BitSet([variable(a),variable(c)])
    state, pr = max_sum_lower_bound(circuit, quer_ac)
    @test state[1,variable(a)] == false && state[1,variable(c)] == true
    @test pr ≈ 0.4 * 0.9
    @test log(pr) ≈ MAR(circuit,state)[1]
end