using Test
using ProbabilisticCircuits
using DataFrames
using BinaryDecisionDiagrams: Diagram, BinaryDecisionDiagrams
const BDD = BinaryDecisionDiagrams
import LogicCircuits: Vtree, respects_vtree

@testset "BDD tests" begin
    # Set up a logic constraint ϕ as a BDD and scope size n. Sample m PSDDs.
    function case(ϕ::Diagram, n::Integer; atol::Real = 1e-3)
        # All possible valuations (including impossible ones).
        M = BDD.all_valuations(collect(1:n))
        # Get only possible worlds.
        W = M[findall(ϕ.(eachrow(M))),:]
        # Assign random probabilities for each world in W.
        R = rand(1:20, size(W, 1))
        # Construct a dataset that maps the distribution of R (world W[i] repeats R[i] times).
        D = DataFrame(vcat([repeat(W[i,:], 1, R[i])' for i ∈ 1:size(W, 1)]...))
        C = learn_bdd(ϕ, D; pseudocount = 0.0)
        T = DataFrame(M)
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

    case(BDD.or(BDD.and(1, 2), BDD.and(3, BDD.:¬(4)), BDD.and(BDD.:¬(1), 5)), 5)
    case(BDD.and(BDD.:→(1, 3), BDD.:→(5, BDD.:¬(2))), 5)
    case(BDD.or(BDD.and(1, 2, 3), BDD.and(4, 5)), 5)
    case(BDD.exactly(3, collect(1:5)), 5)
    case(BDD.atleast(3, collect(1:5)), 5)
    case(BDD.atmost(3, collect(1:5)), 5)
end
