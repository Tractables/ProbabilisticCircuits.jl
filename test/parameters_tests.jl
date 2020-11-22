using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame
using CUDA: CUDA

@testset "MLE tests" begin
    
    # Binary dataset
    dfb = DataFrame(BitMatrix([true false; true true; false true]))
    r = fully_factorized_circuit(ProbCircuit,num_features(dfb))
    
    estimate_parameters(r,dfb; pseudocount=1.0)
    @test log_likelihood_avg(r,dfb) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=1.0)

    estimate_parameters(r,dfb; pseudocount=0.0)
    @test log_likelihood_avg(r,dfb) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=0.0)

    if CUDA.functional()
        # Binary dataset
        dfb_gpu = to_gpu(dfb)
        
        estimate_parameters(r, dfb_gpu; pseudocount=1.0)
        @test log_likelihood_avg(r,dfb_gpu) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=1.0)

        estimate_parameters(r, dfb_gpu; pseudocount=0.0)
        @test log_likelihood_avg(r,dfb_gpu) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=0.0)
    end

end

@testset "Weighted MLE tests" begin
    # Binary dataset
    dfb = DataFrame(BitMatrix([true false; true true; false true]))
    r = fully_factorized_circuit(ProbCircuit,num_features(dfb))
    
    # Weighted binary dataset
    weights = DataFrame(weight = [0.6, 0.6, 0.6])
    wdfb = add_sample_weights(dfb, weights)
    
    dfb = DataFrame(BitMatrix([true false; true true; false true]))
    
    estimate_parameters(r,wdfb; pseudocount=1.0)
    @test log_likelihood_avg(r,dfb) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=1.0)
    @test log_likelihood_avg(r,dfb) ≈ log_likelihood_avg(r,wdfb)
    
    estimate_parameters(r,wdfb; pseudocount=0.0)
    @test log_likelihood_avg(r,dfb) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=0.0)
    @test log_likelihood_avg(r,dfb) ≈ log_likelihood_avg(r,wdfb)

    if CUDA.functional()

        # Binary dataset
        dfb_gpu = to_gpu(dfb)
        
        # Weighted binary dataset
        wdfb_gpu = to_gpu(wdfb)
        
        estimate_parameters(r, wdfb_gpu; pseudocount=1.0)
        @test log_likelihood_avg(r,dfb_gpu) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=1.0)
        @test log_likelihood_avg(r,dfb_gpu) ≈ log_likelihood_avg(r, wdfb_gpu)
        
        estimate_parameters(r, dfb; pseudocount=1.0, use_gpu=true)
        @test log_likelihood_avg(r,dfb_gpu) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=1.0)
        @test log_likelihood_avg(r,dfb_gpu) ≈ log_likelihood_avg(r, wdfb_gpu)
        
        estimate_parameters(r, wdfb_gpu; pseudocount=0.0)
        @test log_likelihood_avg(r,dfb_gpu) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=0.0)
        @test log_likelihood_avg(r,dfb_gpu) ≈ log_likelihood_avg(r, wdfb_gpu)

    end
end

@testset "Batched MLE tests" begin
    # Binary dataset
    dfb = DataFrame(BitMatrix([true false; true true; false true]))
    r = fully_factorized_circuit(ProbCircuit,num_features(dfb))
    
    # Weighted binary dataset
    weights = DataFrame(weight = [0.6, 0.6, 0.6])
    wdfb = add_sample_weights(dfb, weights)
    
    dfb = DataFrame(BitMatrix([true false; true true; false true]))
    
    # Batched dataset
    batched_wdfb = batch(wdfb, 1)
    
    estimate_parameters(r,batched_wdfb; pseudocount=1.0)
    @test log_likelihood_avg(r,dfb) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=1.0)
    @test log_likelihood_avg(r,dfb) ≈ log_likelihood_avg(r,wdfb)
    @test log_likelihood_avg(r,batched_wdfb) ≈ log_likelihood_avg(r,wdfb)
    
    estimate_parameters(r,batched_wdfb; pseudocount=0.0)
    @test log_likelihood_avg(r,dfb) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=0.0)
    @test log_likelihood_avg(r,dfb) ≈ log_likelihood_avg(r,wdfb)
    @test log_likelihood_avg(r,batched_wdfb) ≈ log_likelihood_avg(r,wdfb)

    if CUDA.functional()

        # Binary dataset
        dfb_gpu = to_gpu(dfb)
        
        # Weighted binary dataset
        wdfb_gpu = to_gpu(wdfb)
        
        # Batched dataset
        batched_wdfb_gpu = to_gpu(batch(wdfb, 1))
        
        estimate_parameters(r, batched_wdfb_gpu; pseudocount=1.0)
        @test log_likelihood_avg(r,dfb_gpu) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=1.0)
        @test log_likelihood_avg(r,dfb_gpu) ≈ log_likelihood_avg(r, wdfb_gpu)
        
        estimate_parameters(r, batched_wdfb_gpu; pseudocount=0.0)
        @test log_likelihood_avg(r,dfb_gpu) ≈ LogicCircuits.Utils.fully_factorized_log_likelihood(dfb; pseudocount=0.0)
        @test log_likelihood_avg(r,dfb_gpu) ≈ log_likelihood_avg(r, wdfb_gpu)

    end
end

@testset "Soft MLE test" begin
    # Weighted binary dataset
    dfb = DataFrame(BitMatrix([true false; true true; false false]))
    dfb = soften(dfb, 0.001; scale_by_marginal = false)
    weights = DataFrame(weight = [0.6, 0.6, 0.6])
    wdfb = add_sample_weights(dfb, weights)
    
    # Binary dataset
    dfb = DataFrame(BitMatrix([true false; true true; false false]))
    dfb = soften(dfb, 0.001; scale_by_marginal = false)
    
    r = fully_factorized_circuit(ProbCircuit,num_features(dfb))
    
    estimate_parameters(r, dfb; pseudocount=0.0)
    @test r.children[1].children[1].log_probs[1] ≈ -0.4059652245524093 atol = 1e-6
    @test r.children[1].children[1].log_probs[2] ≈ -1.0976127839931185 atol = 1e-6
    @test r.children[1].children[2].log_probs[1] ≈ -1.0976127839931185 atol = 1e-6
    @test r.children[1].children[2].log_probs[2] ≈ -0.4059652245524093 atol = 1e-6
    
    estimate_parameters(r, wdfb; pseudocount=0.0)
    @test r.children[1].children[1].log_probs[1] ≈ -0.4059652245524093 atol = 1e-6
    @test r.children[1].children[1].log_probs[2] ≈ -1.0976127839931185 atol = 1e-6
    @test r.children[1].children[2].log_probs[1] ≈ -1.0976127839931185 atol = 1e-6
    @test r.children[1].children[2].log_probs[2] ≈ -0.4059652245524093 atol = 1e-6
    
    if CUDA.functional()
        estimate_parameters(r, to_gpu(dfb); pseudocount=0.0)
        @test r.children[1].children[1].log_probs[1] ≈ -0.4059652245524093 atol = 1e-6
        @test r.children[1].children[1].log_probs[2] ≈ -1.0976127839931185 atol = 1e-6
        @test r.children[1].children[2].log_probs[1] ≈ -1.0976127839931185 atol = 1e-6
        @test r.children[1].children[2].log_probs[2] ≈ -0.4059652245524093 atol = 1e-6
        
        estimate_parameters(r, to_gpu(wdfb); pseudocount=0.0)
        @test r.children[1].children[1].log_probs[1] ≈ -0.4059652245524093 atol = 1e-6
        @test r.children[1].children[1].log_probs[2] ≈ -1.0976127839931185 atol = 1e-6
        @test r.children[1].children[2].log_probs[1] ≈ -1.0976127839931185 atol = 1e-6
        @test r.children[1].children[2].log_probs[2] ≈ -0.4059652245524093 atol = 1e-6
    end
    
end

@testset "EM tests" begin
    data = DataFrame([true missing])
    vtree2 = PlainVtree(2, :balanced)
    pc = fully_factorized_circuit(StructProbCircuit, vtree2).children[1]
    uniform_parameters(pc)
    pc.children[1].prime.log_probs .= log.([0.3, 0.7])
    pc.children[1].sub.log_probs .= log.([0.4, 0.6])
    estimate_parameters_em(pc, data; pseudocount=0.0)
    @test all(pc.children[1].prime.log_probs .== log.([1.0, 0.0]))
    @test pc.children[1].sub.log_probs[1] .≈ log.([0.4, 0.6])[1] atol=1e-6
    
    if CUDA.functional()
        data_gpu = to_gpu(data)
        
        # uniform_parameters(pc)
        # estimate_parameters_em(pc, data_gpu; pseudocount=0.0)
        # @test all(pc.children[1].prime.log_probs .== log.([1.0, 0.0]))
        # @test pc.children[1].sub.log_probs[1] .≈ log.([0.4, 0.6])[1] atol=1e-6
    end

    dfb = DataFrame(BitMatrix([true false; true true; false true]))
    r = fully_factorized_circuit(ProbCircuit,num_features(dfb))
    uniform_parameters(r)
    estimate_parameters(r,dfb; pseudocount=1.0)
    paras1 = ParamBitCircuit(r, dfb).params
    uniform_parameters(r)
    estimate_parameters_em(r, dfb; pseudocount=1.0)
    paras2 = ParamBitCircuit(r, dfb).params
    @test all(paras1 .≈ paras2)
end

@testset "Weighted EM tests" begin
    data = DataFrame([true missing])
    weights = DataFrame(weight = [1.0])
    wdata = hcat(data, weights)
    
    vtree2 = PlainVtree(2, :balanced)
    pc = fully_factorized_circuit(StructProbCircuit, vtree2).children[1]
    uniform_parameters(pc)
    pc.children[1].prime.log_probs .= log.([0.3, 0.7])
    pc.children[1].sub.log_probs .= log.([0.4, 0.6])
    estimate_parameters_em(pc, wdata; pseudocount=0.0)
    @test all(pc.children[1].prime.log_probs .== log.([1.0, 0.0]))
    @test pc.children[1].sub.log_probs[1] .≈ log.([0.4, 0.6])[1] atol=1e-6
    
    if CUDA.functional()
        data_gpu = to_gpu(data)
        weights_gpu = to_gpu(weights)
        
        # estimate_parameters_em(pc, data_gpu, weights_gpu; pseudocount=0.0)
        # @test all(pc.children[1].prime.log_probs .== log.([1.0, 0.0]))
        # @test pc.children[1].sub.log_probs[1] .≈ log.([0.4, 0.6])[1] atol=1e-6
    end

    dfb = DataFrame(BitMatrix([true false; true true; false true]))
    weights = DataFrame(weight = [0.6, 0.6, 0.6])
    wdfb = hcat(dfb, weights)
    r = fully_factorized_circuit(ProbCircuit,num_features(dfb))
    uniform_parameters(r)
    estimate_parameters(r,wdfb; pseudocount=1.0)
    paras1 = ParamBitCircuit(r, wdfb).params
    uniform_parameters(r)
    estimate_parameters_em(r, wdfb; pseudocount=1.0)
    paras2 = ParamBitCircuit(r, wdfb).params
    @test all(paras1 .≈ paras2)
end

@testset "Batched EM tests" begin
    dfb = DataFrame(BitMatrix([true false; true true; false true; true true]))
    weights = DataFrame(weight = [0.6, 0.6, 0.6, 0.6])
    wdfb = add_sample_weights(dfb, weights)
    batched_wdfb = batch(wdfb)
    
    r = fully_factorized_circuit(ProbCircuit,num_features(dfb))
    uniform_parameters(r)
    
    estimate_parameters(r, batched_wdfb; pseudocount=1.0)
    paras1 = ParamBitCircuit(r, wdfb).params
    uniform_parameters(r)
    estimate_parameters_em(r, batched_wdfb; pseudocount=1.0)
    paras2 = ParamBitCircuit(r, wdfb).params
    @test all(paras1 .≈ paras2)
end

@testset "Bagging tests" begin
    # Binary dataset
    dfb = DataFrame(BitMatrix([true true; true true; true true; true true]))
    r = fully_factorized_circuit(ProbCircuit,num_features(dfb))
    # bag_dfb = bagging_dataset(dfb; num_bags = 2, frac_examples = 1.0)
    bag_dfb = Array{DataFrame}(undef, 2)
    bag_dfb[1] = dfb[[2, 1, 3, 4], :]
    bag_dfb[2] = dfb[[4, 3, 2, 1], :]
    
    r = compile(SharedProbCircuit, r, 2)
    
    params = estimate_parameters(r, bag_dfb; pseudocount = 1.0)
    @test all(abs.(params[1] .- params[2]) .< 1e-6)
    
    params = estimate_parameters(r, bag_dfb; pseudocount = 2.0)
    @test all(abs.(params[1] .- params[2]) .< 1e-6)
end