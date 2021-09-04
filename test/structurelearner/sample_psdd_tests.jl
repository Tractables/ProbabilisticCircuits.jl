using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames


@testset "SamplePSDD tests" begin
    # Set up a logic constraint ϕ as a BDD and scope size n. Sample m PSDDs.
    function case(ϕ::Bdd, n::Integer; m::Integer = 20, atol::Real = 0)
        # All possible valuations (including impossible ones).
        M = all_valuations(collect(1:n))
        # Get only possible worlds.
        W = M[findall(ϕ.(eachrow(M))),:]
        # Assign random probabilities for each world in W.
        R = rand(1:20, size(W, 1))
        # Construct a dataset that maps the distribution of R (world W[i] repeats R[i] times).
        D = DataFrame(vcat([repeat(W[i,:], 1, R[i])' for i ∈ 1:size(W, 1)]...), :auto)
        # Learn PSDDs from ϕ and D. Overfit them so that we can use ≈ without Julia complaining.
        C = Vector{StructProbCircuit}(undef, m)
        Threads.@threads for i ∈ 1:m
            C[i] = sample_psdd(ϕ, Vtree(n, :random), 16, D; pseudocount = 0.0, maxiter = 100)
        end
        T = DataFrame(M, :auto)
        for i ∈ 1:m
            # Test smoothness.
            @test issmooth(C[i])
            # Test decomposability.
            @test isdecomposable(C[i])
            # Test determinism.
            @test isdeterministic(C[i])
            # Tests if respects vtree.
            @test respects_vtree(C[i], C[i].vtree)
            # Test consistency.
            @test (EVI(C[i], T) .> -Inf) == ϕ.(eachrow(M))
            # Test probabilities.
            evi = exp.(EVI(C[i], T))
            @test isapprox(evi[findall(>(0), evi)], (R/sum(R)); atol)
        end
    end

    case((1 ∧ 2) ∨ (3 ∧ ¬4) ∨ (¬1 ∧ 5), 5)
    case((1 → 3) ∧ (5 → ¬2), 5)
    case((1 ∨ 2 ∨ 3) ∧ (4 ∨ 5), 5)
    case((1 ∧ 2 ∧ 3) ∨ (4 ∧ 5), 5)
    case(exactly(3, collect(1:5)), 5)
    case(atleast(3, collect(1:5)), 5)
    case(atmost(3, collect(1:5)), 5)
end
