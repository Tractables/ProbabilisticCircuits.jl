using Test, ProbabilisticCircuits
# using TikzPictures

include("../helper/plain_dummy_circuits.jl")

@testset "PC plotting" begin
    
    mktempdir() do tmp
            
        # Note: omitting rendering tests to speed up CI

        pc = little_3var()
        p = @test_nowarn plot(pc)
        # @test_nowarn save(SVG("$tmp/example1.svg"), p)

        pc = little_3var_categorical()
        p = @test_nowarn plot(pc)
        # @test_nowarn save(SVG("$tmp/example2.svg"), p)

        pc = little_hybrid_circuit()
        p = @test_nowarn plot(pc)
        # @test_nowarn save(SVG("$tmp/example3.svg"), p)
    
    end
end

