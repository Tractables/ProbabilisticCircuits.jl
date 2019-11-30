using Test
using Statistics
using .Juice
using .Utils

# train_mixture for 2 iterations, with uniformed initial weights, \alpha = 1.0, pseudocount = 1.0
function train_mixture_clt_test()
    iters = 2
    pseudocount = 1.0
    
    dataset2ll = Dict(
        "msnbc" => [-6.539837790664975, -6.54002533295229, -6.540127281851413],
        "plants" => [-16.222515128413086, -16.51758834659623, -16.52399975090857],
        "ad" => [-16.753212802979167, -18.20900862845052, -17.276757736488754],
        "kdd" => [-2.5284795731387364, -2.526484244470829, -2.294883148356537])

    for name in ["msnbc", "plants", "ad", "kdd"]
        data = dataset(twenty_datasets(name); do_shuffle=false, batch_size=-1);
        clt = learn_chow_liu_tree(train(data); α = 1.0, parametered = true);
        pcs = [compile_prob_circuit_from_clt(clt) for i in 1:3];
        mixture = FlatMixture([1/3, 1/3, 1/3], pcs)
        #
        # av TODO: make this more generic
        directory="./"
        log_path = directory * name
        logger, log_option = construct_logger(;log_path=log_path,
                                              train_x=train(data),
                                              valid_x=valid(data),
                                              test_x=test(data), 
            cache_per_example=false, cache_per_ite=true, iters=iters, opts=[])
    
        train_mixture(mixture, convert(XBatches, train(data)), pseudocount, iters; logger = logger)
        @test [log_option.train_ll[1,end], log_option.valid_ll[1,end], log_option.test_ll[1,end]] ≈ dataset2ll[name]
    end
end

# copy from MnistEMBenchmark.jl
# train_mixture for 5 iterations, with uniformed initial weights, pseudocount = 1.0
# slow 
function train_mixture_mnist_test()

    # intermediate results from previous em
    weights_per_ite = #[[0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
        [[0.4607554457051676, 0.19770014903944916, 0.34154440525538327],
        [0.4768764321107725, 0.1992797357651258, 0.3238438321241017],
        [0.47905677892956455, 0.2043911049025035, 0.31655211616793194],
        [0.4658285928167285, 0.22178451166573118, 0.3123868955175404],
        [0.4477936215643856, 0.24660259867989698, 0.30560377975571745]]

    ll_per_ite = #[[-135.08304153401804, -135.4178211752002, -134.376733742139],
        [[-133.69501004313284, -134.2303552285715, -133.14868008100967],
        [-133.16761214283747, -133.74176242188298, -132.65102900767934],
        [-132.64222641450638, -133.21967768290085, -132.12386816645292],
        [-131.94724374558984, -132.50733369134497, -131.43229536261177],
        [-131.25402582740003, -131.80634620543526, -130.771358334033]]

    batch_size = 10000
    iterations = 3

    num_copies_a = 1
    num_copies_b = 1
    num_copies_c = 1 # 

    data = dataset(sampled_mnist(); do_shuffle=false, batch_size=batch_size);

    lc_a = load_smooth_logical_circuit("test/circuits/mnist-antonio.psdd")
    lc_b = load_smooth_logical_circuit("test/circuits/mnist-large.circuit")
    lc_c = load_smooth_logical_circuit("test/circuits/mnist-1M.circuit")

    copy_a(i) = ProbΔ(lc_a);
    copy_b(i) = ProbΔ(lc_b);
    copy_c(i) = ProbΔ(lc_c);

    pcs = Vector{ProbΔ}()
    append!(pcs, [copy_a(i) for i in 1:num_copies_a]);
    append!(pcs, [copy_b(i) for i in 1:num_copies_b]);
    append!(pcs, [copy_c(i) for i in 1:num_copies_c]);

    for pc in pcs
        estimate_parameters(pc, train(data); pseudocount=1.0)
    end

    mixture = FlatMixture([1/3, 1/3, 1/3], pcs);

    log_dir = joinpath(tempname(), "mnist")
    rm(log_dir;recursive=true, force=true)
    log_path = joinpath(log_dir, "tmp.log")
    logger, log_option = construct_logger(;log_path=log_path,
                                          train_x=train(data), valid_x=valid(data), test_x=test(data), 
            cache_per_example=false, cache_per_ite=true, iters=6, opts=[]);

    train_mixture(mixture, train(data), 1.0, 5; logger=logger);

    @test mean(log_option.train_ll) ≈ ll_per_ite[5][1] atol=1.0e-5
    @test mean(log_option.valid_ll) ≈ ll_per_ite[5][2] atol=1.0e-5
    @test mean(log_option.test_ll) ≈ ll_per_ite[5][3] atol=1.0e-5
    rm(log_dir;recursive=true)
end

@testset "EM parameter learner" begin
    train_mixture_mnist_test()
end
