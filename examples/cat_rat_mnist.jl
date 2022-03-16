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


function truncate(data::Matrix; bits)
    (data .- one(UInt32)) .÷ 2^bits .+ one(UInt32)
end

function generate_rat(train)
    # RAT hyperparamters
    num_features = size(train, 2)
    num_nodes_root = 1
    num_nodes_region = 10
    num_nodes_leaf   = 10
    rg_depth        = 4
    rg_replicas     = 10
    input_type = Categorical
    balance_childs_parents = false
    RAT(num_features; num_nodes_region, num_nodes_leaf, rg_depth, rg_replicas, input_type, balance_childs_parents)
end

function run(; batch_size = 256, num_epochs1 = 1, num_epochs2 = 1, num_epochs3 = 20, 
    pseudocount = 0.01, param_inertia1 = 0.2, param_inertia2 = 0.9, param_inertia3 = 0.9)

    train, test = mnist_cpu();
    train_gpu, test_gpu = mnist_gpu();
    trunc_train = truncate(train; bits = 5);

    @info "Generating RAT SPN...."
    @time pc = generate_rat(trunc_train);
    init_parameters(pc; perturbation = 0.4);

    println("Number of free parameters: $(num_parameters(pc))")

    @info "Moving circuit to GPU... "
    CUDA.@time bpc = CuBitsProbCircuit(BitsProbCircuit(pc));

    @show length(bpc.nodes)

    @info "EM"
    softness    = 0
    @time mini_batch_em(bpc, train_gpu, num_epochs1; batch_size, pseudocount, 
    			 softness, param_inertia = param_inertia1, param_inertia_end = param_inertia2)
                     
    @time mini_batch_em(bpc, train_gpu, num_epochs2; batch_size, pseudocount, 
    			 softness, param_inertia = param_inertia2, param_inertia_end = param_inertia3)
    
    for iter=1:num_epochs3
        @info "Iter $iter"
        @time full_batch_em(bpc, train_gpu, 5; batch_size, pseudocount, softness)

        ll3 = loglikelihood(bpc, test_gpu; batch_size)
        println("test LL: $(ll3)")

        @time do_sample(bpc, iter)
    end

    print("update parameters")
    @time ProbabilisticCircuits.update_parameters(bpc);
    return pc, bpc
end

function do_sample(bpc, iter=999)
    CUDA.@time sms = sample(bpc, 100, 28*28, [UInt32]);

    do_img(i) = begin
        img = Array{Float32}(sms[i,1,:]) ./ 256.0
        img = transpose(reshape(img, (28, 28)))
        imresize(colorview(Gray, img), ratio=4)
    end

    arr = [do_img(i) for i=1:size(sms, 1)]
    imgs = mosaicview(arr, fillvalue=1, ncol=10, npad=4)
    save("samples/rat_samples_$(iter).png", imgs) 
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

pc, bpc = run(; batch_size = 128, num_epochs1 = 2, num_epochs2 = 2, num_epochs3 = 2);
# do_sample(bpc)
# try_map(pc, bpc)