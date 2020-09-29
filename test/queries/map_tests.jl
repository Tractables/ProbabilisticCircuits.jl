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

    # same MAP states on CPU and GPU
    cpu_gpu_agree(data_marg) do d 
        MAP(prob_circuit, d)[1]
    end

    # same MAP probabilities on CPU and GPU
    cpu_gpu_agree_approx(data_marg) do d 
        MAP(prob_circuit, d)[2]
    end

end