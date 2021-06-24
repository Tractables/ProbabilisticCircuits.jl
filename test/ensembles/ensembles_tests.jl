using Test
using ProbabilisticCircuits
using DataFrames
using BinaryDecisionDiagrams: Diagram, BinaryDecisionDiagrams
const BDD = BinaryDecisionDiagrams

@testset "ensemble tests with SamplePSDD" begin
    # Set up a logic constraint ϕ as a BDD and scope size n. Sample m PSDDs.
    function case(ϕ::Diagram, n::Integer, strategy::Symbol; m::Integer = 20, atol::Real = 1e-2)::Ensemble{StructProbCircuit}
        # All possible valuations (including impossible ones).
        M = BDD.all_valuations(collect(1:n))
        # Get only possible worlds.
        W = M[findall(ϕ.(eachrow(M))),:]
        # Assign random probabilities for each world in W.
        R = rand(1:20, size(W, 1))
        # Construct a dataset that maps the distribution of R (world W[i] repeats R[i] times).
        D = DataFrame(vcat([repeat(W[i,:], 1, R[i])' for i ∈ 1:size(W, 1)]...))
        # Learn ensemble of PSDDs from ϕ and D.
        E = ensemble_sample_psdd(m, ϕ, 16, D; strategy, pseudocount = 0.0, maxiter = 100,
                                 verbose = false, vtree_bias = 0.8)
        T = DataFrame(M)
        # Test consistency.
        @test (EVI(E, T) .> -Inf) == ϕ.(eachrow(M))
        # Test probabilities.
        evi = exp.(EVI(E, T))
        @test isapprox(evi[findall(>(0), evi)], (R/sum(R)); atol)
        return E
    end

    Es = Vector{Ensemble{StructProbCircuit}}()
    for strategy ∈ [:likelihood, :uniform, :em, :stacking]
        push!(Es, case(BDD.or(BDD.and(1, 2), BDD.and(3, BDD.:¬(4)), BDD.and(BDD.:¬(1), 5)), 5, strategy))
        push!(Es, case(BDD.and(BDD.:→(1, 3), BDD.:→(5, BDD.:¬(2))), 5, strategy))
        push!(Es, case(BDD.or(BDD.and(1, 2, 3), BDD.and(4, 5)), 5, strategy))
        push!(Es, case(BDD.exactly(3, collect(1:5)), 5, strategy))
        push!(Es, case(BDD.atleast(3, collect(1:5)), 5, strategy))
        push!(Es, case(BDD.atmost(3, collect(1:5)), 5, strategy))
    end

    tmp = mktempdir()
    @testset "Saving and loading ensembles" begin
        for (i, E) ∈ enumerate(Es)
            @test_nowarn save_as_ensemble("$tmp/$i.esbl", E; quiet = true)
        end
    end
    Rs = Vector{Ensemble{StructProbCircuit}}()
    T = DataFrame(BDD.all_valuations(1:5))
    @testset "Loading ensembles" begin
        for i ∈ 1:length(Es)
            E = load_as_ensemble("$tmp/$i.esbl"; quiet = true)
            @test EVI(E, T) ≈ EVI(Es[i], T)
        end
    end
end
