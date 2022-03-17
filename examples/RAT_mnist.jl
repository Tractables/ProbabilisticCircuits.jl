using ProbabilisticCircuits
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, loglikelihoods, full_batch_em, mini_batch_em
using MLDatasets
using CUDA
using Images

# device!(collect(devices())[2])

function mnist_cpu()
    train_int = transpose(reshape(MNIST.traintensor(UInt8), 28*28, :));
    test_int = transpose(reshape(MNIST.testtensor(UInt8), 28*28, :));

    train_cpu = UInt32.(train_int) .+ one(UInt32);
    test_cpu = UInt32.(test_int) .+ one(UInt32);

    train_cpu, test_cpu
end

function mnist_gpu()
    cu.(mnist_cpu())
end

function run(; batch_size = 256, num_epochs1 = 1, num_epochs2 = 1, num_epochs3 = 20, 
    pseudocount = 0.01, param_inertia1 = 0.2, param_inertia2 = 0.9, param_inertia3 = 0.9)

    train, test = mnist_cpu();
    train_gpu, test_gpu = mnist_gpu();

    @info "Generating RAT SPN...."
    num_nodes_root = 1
    num_nodes_region = 20
    num_nodes_leaf   = 20
    rg_depth        = 4
    rg_replicas     = 20

    input_func = RAT_InputFunc(Binomial, 256);  
    # input_func = RAT_InputFunc(Categorical, 256);
    # input_func(var) =
    #     summate([InputNode(var, Binomial(256)) for i=1:2])

    @show num_nodes_region
    @show num_nodes_leaf
    @show rg_depth
    @show rg_replicas

    num_features = size(train, 2)
    @time pc = RAT(num_features; input_func, num_nodes_region, num_nodes_leaf, rg_depth, rg_replicas, balance_childs_parents=false);
    init_parameters(pc; perturbation = 0.4);    

    @time println("Number of free parameters: $(num_parameters(pc))")

    @info "Moving circuit to GPU... "
    CUDA.@time bpc = CuBitsProbCircuit(BitsProbCircuit(pc));
    @show length(bpc.nodes)

    @info "Mini EM 1"
    softness    = 0
    @time mini_batch_em(bpc, train_gpu, num_epochs1; batch_size, pseudocount, 
    			 softness, param_inertia = param_inertia1, param_inertia_end = param_inertia2)

    ll1 = loglikelihood(bpc, test_gpu; batch_size)
    println("test LL: $(ll1)")

    @info "Mini EM 2"                 
    @time mini_batch_em(bpc, train_gpu, num_epochs2; batch_size, pseudocount, 
    			 softness, param_inertia = param_inertia2, param_inertia_end = param_inertia3)

    ll2 = loglikelihood(bpc, test_gpu; batch_size)
    println("test LL: $(ll2)")
    
    @info "Full EM"
    for iter=1:num_epochs3
        @info "Iter $iter"
        @time full_batch_em(bpc, train_gpu, 5; batch_size, pseudocount, softness)

        ll3 = loglikelihood(bpc, test_gpu; batch_size)
        println("test LL: $(ll3)")

        do_sample(bpc, iter)
    end

    @info "Update parameters pbc -> pc"
    @time ProbabilisticCircuits.update_parameters(bpc);
    return pc, bpc
end

function do_sample(cur_pc, iter)
    file_name = "samples/rat_samples_$(iter).png"
    @info "Sampling $(file_name)"

    if cur_pc isa CuBitsProbCircuit
        sms = sample(cur_pc, 100, 28*28, [UInt32]);
    elseif cur_pc isa ProbCircuit
        sms = sample(cur_pc, 100, [UInt32]);
    end

    do_img(i) = begin
        img = Array{Float32}(sms[i,1,1:28*28]) ./ 256.0
        img = transpose(reshape(img, (28, 28)))
        imresize(colorview(Gray, img), ratio=4)
    end

    @time begin
        arr = [do_img(i) for i=1:size(sms, 1)]
        imgs = mosaicview(arr, fillvalue=1, ncol=10, npad=4)
        save(file_name, imgs) 
    end
end

function try_map(pc, bpc)
    @info "MAP"
    train_gpu, _ = mnist_gpu();
    data = Array{Union{Missing, UInt32}}(train_gpu[1:10, :]);
    data[:, 1:400] .= missing;
    data_gpu = cu(data);

    # @time MAP(pc, data; batch_size=10)
    MAP(bpc, data_gpu; batch_size=10)
end

pc, bpc = run(; batch_size = 1000, num_epochs1 = 10, num_epochs2 = 10, num_epochs3 = 100);
# do_sample(bpc, 999)
# try_map(pc, bpc)