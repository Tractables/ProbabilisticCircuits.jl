using Test
using ProbabilisticCircuits
using DataFrames
using BinaryDecisionDiagrams: Diagram, BinaryDecisionDiagrams
const BDD = BinaryDecisionDiagrams
import LogicCircuits: Vtree, respects_vtree

@testset "SamplePSDD tests" begin
    # Set up a logic constraint ϕ as a BDD and scope size n. Sample m PSDDs.
    function case(ϕ::Diagram, n::Integer; m::Integer = 20, atol::Real = 0)
        # All possible valuations (including impossible ones).
        M = BDD.all_valuations(collect(1:n))
        # Get only possible worlds.
        W = M[findall(ϕ.(eachrow(M))),:]
        # Assign random probabilities for each world in W.
        R = rand(1:20, size(W, 1))
        # Construct a dataset that maps the distribution of R (world W[i] repeats R[i] times).
        D = DataFrame(vcat([repeat(W[i,:], 1, R[i])' for i ∈ 1:size(W, 1)]...))
        # Learn PSDDs from ϕ and D. Overfit them so that we can use ≈ without Julia complaining.
        C = Vector{StructProbCircuit}(undef, m)
        Threads.@threads for i ∈ 1:m
            C[i] = sample_psdd(ϕ, Vtree(n, :random), 16, D; pseudocount = 0.0, maxiter = 100)
        end
        T = DataFrame(M)
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

    case(BDD.or(BDD.and(1, 2), BDD.and(3, BDD.:¬(4)), BDD.and(BDD.:¬(1), 5)), 5)
    case(BDD.and(BDD.:→(1, 3), BDD.:→(5, BDD.:¬(2))), 5)
    case(BDD.or(BDD.and(1, 2, 3), BDD.and(4, 5)), 5)
    case(BDD.exactly(3, collect(1:5)), 5)
    case(BDD.atleast(3, collect(1:5)), 5)
    case(BDD.atmost(3, collect(1:5)), 5)
end
