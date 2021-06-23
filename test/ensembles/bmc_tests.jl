using Test
using ProbabilisticCircuits
using DataFrames
using BinaryDecisionDiagrams

@testset "BMC tests with SamplePSDD" begin
    # Set up a logic constraint ϕ as a BDD and scope size n.
    function case(ϕ::Diagram, n::Integer; atol::Real = 0)
        # All possible valuations (including impossible ones).
        M = all_valuations(collect(1:n))
        # Get only possible worlds.
        W = M[findall(ϕ.(eachrow(M))),:]
        # Assign random probabilities for each world in W.
        R = rand(1:20, size(W, 1))
        # Construct a dataset that maps the distribution of R (world W[i] repeats R[i] times).
        D = DataFrame(vcat([repeat(W[i,:], 1, R[i])' for i ∈ 1:size(W, 1)]...))
        # Learn ensemble of PSDDs from ϕ and D.
        B = bmc_sample_psdd(5, ϕ, 16, D, 3, 4; pseudocount = 0.0, maxiter = 100, verbose = false)
        T = DataFrame(M)
        # Test consistency.
        @test (EVI(B, T) .> -Inf) == ϕ.(eachrow(M))
        # Test probabilities.
        evi = exp.(EVI(B, T))
        @test isapprox(evi[findall(>(0), evi)], (R/sum(R)); atol)
    end

    case((1 ∧ 2) ∨ (3 ∧ ¬4) ∨ (¬1 ∧ 5), 5)
    case((1 → 3) ∧ (5 → ¬2), 5)
    case(and(1, 2, 3) ∨ and(4, 5), 5)
    case(exactly(3, collect(1:5)), 5)
    case(atleast(3, collect(1:5)), 5)
    case(atmost(3, collect(1:5)), 5)
end
