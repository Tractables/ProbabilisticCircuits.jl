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
    end
    
    bern = PlainInputNode(42, BernoulliDist(log(0.1)))
    bit_bern = bits(bern, heap)
    @test isbits(bit_bern)

    cat = PlainInputNode(42, CategoricalDist(6))
    bit_cat = bits(cat, heap)
    @test isbits(bit_cat)
    @test length(heap) == 6

    T = Union{typeof(bit_bern), typeof(bit_cat), typeof(bit_lit)}
    @test Base.isbitsunion(T)
    @test Base.isbitsunion(eltype(T[bit_bern, bit_cat, bit_lit]))

end