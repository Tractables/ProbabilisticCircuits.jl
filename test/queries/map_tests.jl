using Test
using LogicCircuits
using ProbabilisticCircuits
using DataFrames: DataFrame
using CUDA

include("../helper/gpu.jl")

@testset "MAP" begin
    prob_circuit = zoo_psdd("little_4var.psdd");

    data_full = generate_data_all(num_variables(prob_circuit))

    map, mappr = MAP(prob_circuit, data_full)

    @test map == data_full
    
    evipr = EVI(prob_circuit, data_full)
    @test mappr ≈ evipr atol=1e-6

    data_marg = DataFrame([false false false false; 
                      false true true false; 
                      false false true true;
                      false false false missing; 
                      missing true false missing; 
                      missing missing missing missing; 
                      false missing missing missing])

    map, mappr = MAP(prob_circuit, data_marg)

    @test all(zip(eachcol(map), eachcol(data_marg))) do (cf,cm) 
        all(zip(cf, cm)) do (f,m) 
            ismissing(m) || f == m
        end
    end

    mar = MAR(prob_circuit, data_marg)

    @test all(mar .> mappr .- 1e-6)

    map, mappr = MAP(prob_circuit, false, false, false, missing)
    @test map == [false, false, false, true]
    @test mappr ≈ -1.2729657

    # same MAP states on CPU and GPU
    cpu_gpu_agree(data_full) do d 
        MAP(prob_circuit, d)[1]
    end

    # same MAP probabilities on CPU and GPU
    cpu_gpu_agree_approx(data_full) do d 
        MAP(prob_circuit, d)[2]
    end

    # same MAP states on CPU and GPU
    cpu_gpu_agree(data_marg) do d 
        MAP(prob_circuit, d)[1]
    end

    # same MAP probabilities on CPU and GPU
    cpu_gpu_agree_approx(data_marg) do d 
        MAP(prob_circuit, d)[2]
    end

end

@testset "MAP upward pass" begin
    a,b = pos_literals(ProbCircuit, 2)
    circuit = 0.6 * (a * (.5 * b + .5 * -b)) + .4 * (-a * (0.9 * b + .1 * -b))
    data_marg = DataFrame([missing missing])
    map, mappr = MAP(circuit, data_marg)
    data_marg = DataFrame([true true; true false; false true; false false])
    mar = MAR(circuit, data_marg)
    @test all(mar .> mappr .- 1e-6)
end