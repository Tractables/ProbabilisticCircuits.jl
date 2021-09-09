using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames

@testset "BDD tests" begin
    # Set up a logic constraint ϕ as a BDD and scope size n. Sample m PSDDs.
    function case(ϕ::Bdd, n::Integer; atol::Real = 1e-3)
        # All possible valuations (including impossible ones).
        M = all_valuations(collect(1:n))
        # Get only possible worlds.
        W = M[findall(ϕ.(eachrow(M))),:]
        # Assign random probabilities for each world in W.
        R = rand(1:20, size(W, 1))
        # Construct a dataset that maps the distribution of R (world W[i] repeats R[i] times).
        D = DataFrame(vcat([repeat(W[i,:], 1, R[i])' for i ∈ 1:size(W, 1)]...), :auto)
        C = learn_bdd(ϕ, D; pseudocount = 0.0)
        T = DataFrame(M, :auto)
        # Test smoothness.
        @test issmooth(C)
        # Test decomposability.
        @test isdecomposable(C)
        # Test determinism.
        @test isdeterministic(C)
        # Tests if respects vtree.
        @test respects_vtree(C, C.vtree)
        # Test consistency.
        @test (EVI(C, T) .> -Inf) == ϕ.(eachrow(M))
    end
    case((1 ∧ 2) ∨ (3 ∧ ¬4) ∨ (¬1 ∧ 5), 5)
    case((1 → 3) ∧ (5 → ¬2), 5)
    case((1 ∧ 2 ∧ 3) ∨ (4 ∧ 5), 5)
    case(exactly(3, collect(1:5)), 5)
    case(atleast(3, collect(1:5)), 5)
    case(atmost(3, collect(1:5)), 5)
end
