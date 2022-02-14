using Test, ProbabilisticCircuits
using ProbabilisticCircuits: bits, PlainInputNode

@testset "input distributions" begin

    n = PlainInputNode(1, LiteralDist(true))
    @test issetequal(randvars(n), [1])

    n = PlainInputNode(1, BernoulliDist(log(0.5)))
    @test issetequal(randvars(n), [1])
    @test n.dist.logp ≈ log(0.5)

    n = PlainInputNode(1, CategoricalDist(4))
    @test issetequal(randvars(n), [1])
    @test all(n.dist.logps .≈ [log(0.25), log(0.25), log(0.25), log(0.25)])
    
end

@testset "bit input nodes" begin
    
    heap = Float32[]
    bit_lit = nothing

    for sign in [true, false]
        lit = PlainInputNode(42, LiteralDist(sign))
        bit_lit = bits(lit, heap)
        @test isbits(bit_lit)
        @test loglikelihood(dist(bit_lit), false) ≈ log(!sign)
        @test loglikelihood(dist(bit_lit), true) ≈ log(sign)
    end
    
    bern = PlainInputNode(42, BernoulliDist(log(0.1)))
    bit_bern = bits(bern, heap)
    @test isbits(bit_bern)
    @test loglikelihood(dist(bit_bern), 1, heap) ≈ log(0.1)
    @test loglikelihood(dist(bit_bern), 0, heap) ≈ log(1-0.1)

    heap = Float32[]
    cat = PlainInputNode(42, CategoricalDist(6))
    bit_cat = bits(cat, heap)
    @test isbits(bit_cat)
    @test length(heap) == 2*6
    for i = 1:6
        @test loglikelihood(dist(bit_cat), i, heap) ≈ log(1/6)
    end

    T = Union{typeof(bit_bern), typeof(bit_cat), typeof(bit_lit)}
    @test Base.isbitsunion(T)
    @test Base.isbitsunion(eltype(T[bit_bern, bit_cat, bit_lit]))

end