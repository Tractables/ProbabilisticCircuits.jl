using Test
using Random
using LogicCircuits
using ProbabilisticCircuits

@testset "Expectation Maximization algorithm test" begin
    Random.seed!(1337)
    data = dataset(twenty_datasets("nltcs"); do_shuffle=false, batch_size=-1)
    train_x = train(data)

    pc1 = zoo_psdd("nltcs.clt.psdd")
    pc2 = zoo_psdd("nltcs.10split.psdd")
    pcs = [pc1, pc2]
    mf = train_mixture(pcs, convert(XBatches, train_x), 1.0, 3)

    @test mf isa FlatMixtureWithFlow
    @test mf.origin isa FlatMixture
    @test components(mf) == pcs
    @test component_weights(mf) ≈ [0.41093382427863423, 0.5890661757213657] atol=1.0e-9
    @test log_likelihood(mf, convert(XBatches, train_x)) / num_examples(train_x) ≈ -6.475868465920385 atol=1.0e-9
end
