if endswith(@__FILE__, PROGRAM_FILE)
    # this file is run as a script
    include("../../../src/Juice/Juice.jl")
 end

using Test
using .Juice

"""
Given some missing values generates all possible fillings
"""
function generate_all(row::Array{Int8})
    miss_count = count(row .== -1)
    lits = length(row)
    result = Bool.(zeros(1 << miss_count, lits))

    if miss_count == 0
        result[1, :] = copy(row)
    else
        for mask = 0: (1<<miss_count) - 1
            cur = copy(row)
            cur[row .== -1] = transpose(parse.(Bool, split(bitstring(mask)[end-miss_count+1:end], "")))
            result[mask+1,:] = cur
        end
    end
    result
end


function test_expectation_brute_force(pc::ProbΔ, lc::LogisticΔ, data::XData, CLASSES::Int)
    EPS = 1e-7;
    COUNT = size(data.x)[1]
    # Compute True expectation brute force
    true_exp = zeros(COUNT, CLASSES)
    for i in 1:COUNT
        row = data.x[i, :]
        cur_data_all = XData(generate_all(row))

        fc1, calc_p = log_likelihood_per_instance(pc, cur_data_all)
        calc_p = exp.(calc_p)

        fc2, calc_f = class_conditional_likelihood_per_instance(lc, CLASSES, cur_data_all)
        true_exp[i, :] = sum(calc_p .* calc_f, dims=1)
        true_exp[i, :] ./= sum(calc_p) #p_observed
    end

    # Compute Circuit Expect
    calc_exp, cache = Expectation(pc, lc, data);
    for i = 1:COUNT
        for j = 1:CLASSES
            @test true_exp[i,j] ≈ calc_exp[i,j] atol= EPS;
        end
    end    
    # Compute Bottom Up Expectation
    calc_exp_2, exp_flow = ExpectationUpward(pc, lc, data);
    for i = 1:COUNT
        for j = 1:CLASSES
            @test true_exp[i,j] ≈ calc_exp_2[i,j] atol= EPS;
        end
    end
end


@testset "Expectation Brute Force Test Small (4 Var)" begin
    vtree_file      = "test/circuits/little_4var.vtree"
    psdd_file       = "test/circuits/little_4var.psdd"
    logistic_file   = "test/circuits/little_4var.circuit";
    CLASSES = 2
    N = 4

    pc = load_prob_circuit(psdd_file);
    lc = load_logistic_circuit(logistic_file, CLASSES);
    data = XData(Int8.([
                        0 0 0 0; 
                        0 1 1 0; 
                        0 0 1 1;
                        -1 -1 -1 -1;
                        -1 0 1 -1;
                        0 1 -1 1;
                        1 -1 0 -1;
                        -1 0 1 -1;
                        -1 -1 0 1;
                        -1 -1 -1 1;
                        -1 -1 -1 0;
                        ]));

    test_expectation_brute_force(pc, lc, data, CLASSES)
end


@testset "Expectation Brute Force Test Big (15 Var)" begin
    vtree_file      = "test/circuits/expectation/exp-D15-N1000-C4.vtree"
    psdd_file       = "test/circuits/expectation/exp-D15-N1000-C4.psdd"
    logistic_file   = "test/circuits/expectation/exp-D15-N1000-C4.circuit";
    CLASSES = 4
    N = 15
    COUNT = 10

    pc = load_prob_circuit(psdd_file);
    lc = load_logistic_circuit(logistic_file, CLASSES);
    data = XData(Int8.(rand( (-1,0,1), (COUNT, N) )))
    
    test_expectation_brute_force(pc, lc, data, CLASSES)
end