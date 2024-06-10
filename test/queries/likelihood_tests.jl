using Test, DirectedAcyclicGraphs, ProbabilisticCircuits, CUDA
using ProbabilisticCircuits: CuBitsProbCircuit
using StatsFuns

include("../helper/plain_dummy_circuits.jl")
include("../helper/data.jl")

@testset "likelihood" begin
    EPS = 1e-6

    little4var = little_4var();
    @test little4var isa ProbCircuit

    # Step 1.
    data = Matrix{Bool}([0 0 0 0; 0 1 1 0; 0 0 1 1])
    true_probs = [0.07; 0.03; 0.13999999999999999]

    # Bigger Batch size
    probs = exp.(loglikelihoods(little4var, data; batch_size = 32))
    @test true_probs ≈ probs atol=EPS

    # Smaller Batch size
    lls = exp.(loglikelihoods(little4var, data; batch_size = 2))
    @test true_probs ≈ probs atol=EPS

    # Step 2. Add up all probabilities
    @test num_randvars(little4var) == 4
    data_all = generate_data_all(num_randvars(little4var))

    lls_all = loglikelihoods(little4var, data_all; batch_size=16)
    probs_all = exp.(lls_all)
    @test 1.00 ≈ sum(probs_all) atol=EPS

    if CUDA.functional()
        little4var_bpc = CuBitsProbCircuit(little4var)
        data_all_gpu = cu(data_all)

        lls_all_gpu = loglikelihoods(little4var_bpc, data_all_gpu; batch_size=16)
        @test Array(lls_all_gpu) ≈ lls_all atol=EPS
    end

    # GPU Tests Part 2
    if CUDA.functional()
    
        pc = little_3var()
        bpc = CuBitsProbCircuit(pc)

        data = cu([true true false; false true false; false false false])

        lls = Array(loglikelihoods(bpc, data; batch_size = 32))
        avg_ll = loglikelihood(bpc, data; batch_size = 32)
        
        @test lls[1] ≈ log(Float32(0.125))
        @test lls[2] ≈ log(Float32(0.125))
        @test lls[3] ≈ log(Float32(0.125))
        @test avg_ll ≈ log(Float32(0.125))

        pc = little_3var_bernoulli(; p = Float32(0.6))
        bpc = CuBitsProbCircuit(pc)

        lls = Array(loglikelihoods(bpc, data; batch_size = 32))

        @test lls[1] ≈ log(Float32(0.6 * 0.6 * 0.4))
        @test lls[2] ≈ log(Float32(0.4 * 0.6 * 0.4))
        @test lls[3] ≈ log(Float32(0.4 * 0.4 * 0.4))

        data = cu(UInt32.([2 3 4; 5 1 2; 3 4 5]))

        pc = little_3var_categorical(; num_cats = UInt32(5))
        bpc = CuBitsProbCircuit(pc)

        lls = Array(loglikelihoods(bpc, data; batch_size = 32))

        @test lls[1] ≈ log(Float32(0.2^3))

    end

end


@testset "gaussian 1-var-gmm likelihood" begin
    EPS = 1e-6

    pc = little_gmm();
    @test pc isa ProbCircuit

    data = Vector{Float64}([-1.0; 0.0; 1.0])

    gmm_mu = [-1.0, 1.0]
    gmm_sigma = [1.0, 1.0]
    gmm_w = [0.5, 0.5]

    n = size(data, 1) # Num data
    m = size(gmm_w, 1)  # Num mixture components
    
    # Repeat data for each comp. and standardize
    z = ((repeat(reshape(data, 1, n), m, 1)) .- repeat(gmm_mu, 1, n)) ./ repeat(gmm_sigma, 1, n)
    
    # Compute true probs with StatsFuns
    p_m = normpdf.(z)

    # Weighted sum of probs from each dist for each datapoint
    true_probs = transpose(gmm_w) * p_m
    true_probs = reshape(true_probs, n, 1)

    # Bigger Batch size
    probs = exp.(loglikelihoods(pc, reshape(data, n, 1); batch_size = 32))
    probs = reshape(probs, n, 1)

    @test true_probs ≈ probs atol=EPS

    # Smaller Batch size
    lls = exp.(loglikelihoods(pc, reshape(data, n, 1); batch_size = 2))
    @test true_probs ≈ probs atol=EPS

    @test num_randvars(pc) == 1
 
    # GPU Tests
    # TODO: test on GPU
    if CUDA.functional()
        pc = little_gmm()
        bpc = CuBitsProbCircuit(pc)
 
        data = cu(reshape(data, n, 1))

        probs = loglikelihoods(bpc, data; batch_size = 32)
        
        @test true_probs ≈ probs atol=EPS
     end

end


@testset "gaussian 2-var-gmm likelihood" begin
    EPS = 1e-6

    pc = little_2var_gmm();
    @test pc isa ProbCircuit

    data = Matrix{Float64}([-2.0 -2.0; 0.0 0.0; 1.0 1.0; -2.0 0.0])

    gmm_mu = [-2.0 -2.0; 0.0 0.0]
    
    gmm_sigma = 1.0
    gmm_w = [0.2; 0.8]

    n = size(data, 1) # Num data
    m = size(gmm_w, 1)  # Num mixture components
    d = 2
    
    # Compute GMM probs for each data-point
    # z = ((repeat(reshape(data, n, d), 1, 1, m)) .- repeat(reshape(gmm_mu, 1, d, m), n, 1, 1)) ./ repeat([gmm_sigma], n, d, m)
    true_probs = zeros(n)
    for i in 1:n
        for k in 1:m
            x = data[i, :]

            m_mu = gmm_mu[k, :]
            
            # Standardization
            z = (x .- m_mu) ./ gmm_sigma

            # Iterative weighted sum of each comp. probs
            m_w = gmm_w[k]
            true_probs[i] += m_w * prod(normpdf.(z))
        end
    end

    true_probs = reshape(true_probs, n, 1)

    # Bigger Batch size
    probs = exp.(loglikelihoods(pc, reshape(data, n, d); batch_size = 32))
    probs = reshape(probs, n, 1)

    @test true_probs ≈ probs atol=EPS

    # Smaller Batch size
    lls = exp.(loglikelihoods(pc, reshape(data, n, d); batch_size = 2))
    @test true_probs ≈ probs atol=EPS

    @test num_randvars(pc) == 2
 
    # GPU Tests
    # TODO: test on GPU
    if CUDA.functional()
        pc = little_gmm()
        bpc = CuBitsProbCircuit(pc)
 
        data = cu(reshape(data, n, d))

        probs = loglikelihoods(bpc, data; batch_size = 32)
        
        @test true_probs ≈ probs atol=EPS
     end

end