using ProbabilisticCircuits
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, loglikelihoods, full_batch_em, mini_batch_em
using MLDatasets
using CUDA
using Images

device!(collect(devices())[2])

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

function run()
    train, test = mnist_cpu();
    train_gpu, test_gpu = mnist_gpu();
    trunc_train = truncate(train; bits = 5);

    # println("Generating HCLT structure with $latents latents... ");
    # @time pc = hclt(trunc_train[1:5000,:], latents; num_cats = 256, pseudocount = 0.1, input_type = CategoricalDist);
    # init_parameters(pc; perturbation = 0.4);
    print("Generating RAT SPN....")
    @time pc = generate_rat(trunc_train);
    init_parameters(pc; perturbation = 0.4);

    println("Number of free parameters: $(num_parameters(pc))")

    print("Moving circuit to GPU... ")
    CUDA.@time bpc = CuBitsProbCircuit(BitsProbCircuit(pc));

    batch_size  = 2048
    pseudocount = 0.01
    softness    = 0
    epochs_1      = 5
    epochs_2      = 5
    epochs_3      = 10
    @time mini_batch_em(bpc, train_gpu, epochs_1; batch_size, pseudocount, 
    			 softness, param_inertia = 0.2, param_inertia_end = 0.9)
                     
    @time mini_batch_em(bpc, train_gpu, epochs_2; batch_size, pseudocount, 
    			 softness, param_inertia = 0.9, param_inertia_end = 0.95)
    
    @time full_batch_em(bpc, train_gpu, epochs_3; batch_size, pseudocount, softness)

    print("update parameters")
    @time ProbabilisticCircuits.update_parameters(bpc);
    print("Save to file")
    @time write("rat_cat.jpc.gz", pc);
    return circuit, bpc
end

function do_sample(bpc)
    CUDA.@time sms = sample(bpc, 100, 28*28, [UInt32]);

    do_img(i) = begin
        img = Array{Float32}(sms[i,1,:]) ./ 256.0
        img = transpose(reshape(img, (28, 28)))
        imresize(colorview(Gray, img), ratio=4)
    end

    arr = [do_img(i) for i=1:size(sms, 1)]
    imgs = mosaicview(arr, fillvalue=1, ncol=10, npad=4)
    save("samples.png", imgs) 
end

# circuit, bpc = run();
#do_sample(bpc)