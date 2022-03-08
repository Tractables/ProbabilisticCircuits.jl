using CUDA
using ProbabilisticCircuits
import ProbabilisticCircuits as PCs
using ProbabilisticCircuits: BitsProbCircuit, CuBitsProbCircuit, loglikelihoods, full_batch_em, mini_batch_em
using ProbabilisticCircuits: PlainInputNode, PlainSumNode, PlainMulNode
using ProbabilisticCircuits: BitsInput, BitsCategorical
using MLDatasets
using Images

device!(collect(devices())[2])


function mnist_cpu()
    train_x = collect(transpose(reshape(MNIST.traintensor(UInt8), 28*28, :)))
    test_x = collect(transpose(reshape(MNIST.testtensor(UInt8), 28*28, :)))

    train_y = Array{UInt32}(MNIST.trainlabels())
    test_y  = Array{UInt32}(MNIST.testlabels())

    train_x, train_y, test_x, test_y
end

function mnist_gpu()
    cu.(mnist_cpu())
end

function truncate(data::Matrix; bits)
    data .÷ 2^bits
end


function run(; batch_size = 512, pseudocount = 0.1, latents = 16)
    
    train_x, train_y, test_x, test_y = mnist_cpu();

    class_var = 28*28 + 1
    train = hcat(train_x, train_y)
    test  = hcat(test_x, test_y)
   
    #### Learning HCLT
    # train_x_gpu, train_y_gpu, test_x_gpu, test_y_gpu = mnist_gpu();
    # trunc_train = cu(truncate(train_x; bits = 4));
    # println("Generating HCLT structure with $latents latents... ");
    # @time pc = hclt(trunc_train[1:1000,:], latents; num_cats = 256, pseudocount = 0.1, input_type = Categorical);
    # init_parameters(pc; perturbation = 0.4);
    # println("Free parameters: $(num_parameters(pc))");
    # println("Nodes: $(num_nodes(pc))");
    # println("Edges: $(num_edges(pc))");
    
    # TODO: This did not work
    # @time circuit = PlainSumNode([PlainMulNode([PlainInputNode(class_var, Indicator{UInt32}(UInt32(c))), 
    #                                       deepcopy(pc)]) for c=0:9]);

    # @time circuit = PlainSumNode([PlainMulNode([PlainInputNode(class_var, Categorical(10)), 
    #                                       deepcopy(pc)]) for c=1:10]);

    @info "Making Naive Bayes Circuit P(x, y)"
    @time begin
        childs = Vector{ProbCircuit}()
        for c=0:2
            cur_c = PlainInputNode(class_var, Indicator{UInt32}(c))
            prod_x = PlainMulNode([PlainInputNode(i, Categorical(256)) for i=1:28*28])
            push!(childs, PlainMulNode([cur_c, prod_x]))
        end
        circuit = PlainSumNode(childs)
    end

    @info "Moving circuit and data to GPU"
    CUDA.@time begin
        bpc = CuBitsProbCircuit(circuit)
        cutrain = cu(train)
        cutest = cu(test)
    end    

    return circuit, bpc, cutrain, cutest
end

function train(circuit, bpc, cutrain, cutest; batch_size = 512, num_epochs1 = 1, num_epochs2 = 1, num_epochs3 = 1, 
    pseudocount = 0.1, param_inertia1 = 0.2, param_inertia2 = 0.9, param_inertia3 = 0.95)

    
    @info " PCs.clear_input_node_mem"
    softness    = 0
    PCs.clear_input_node_mem(bpc; rate = 0, debug=true)

    # @info " mini_batch_em"
    # @time mini_batch_em(bpc, cutrain, num_epochs1; batch_size, pseudocount, 
    # 			 softness, param_inertia = param_inertia1, param_inertia_end = param_inertia2, debug = false)

    # @device_code_warntype loglikelihoods(bpc, cutest[1:10, :]; batch_size=10)
    # println("test LL: $(ll1)")

    # @time mini_batch_em(bpc, cutrain, num_epochs2; batch_size, pseudocount, 
    # softness, param_inertia = param_inertia2, param_inertia_end = param_inertia3)

    # ll2 = loglikelihood(bpc, cutest; batch_size)
    # println("test LL: $(ll2)")

    # @time full_batch_em(bpc, cutrain, num_epochs3; batch_size, pseudocount, softness)

    # ll3 = loglikelihood(bpc, cutest; batch_size)
    # println("test LL: $(ll3)")

    # print("update parameters")
    # @time ProbabilisticCircuits.update_parameters(bpc)			 
    # @time write("mnist_classify.jpc.gz", circuit);

    nothing
end

function do_sample(bpc, class=2)
    data = CuMatrix{Union{Missing, UInt32}}([i<28*28+1 ? missing : UInt32(class) for j=1:1, i=1:1+28*28]);
    CUDA.@time sms = sample(bpc, 100, data);

    do_img(i) = begin
        img = Array{Float32}(sms[i,1,1:28*28]) ./ 256.0
        img = transpose(reshape(img, (28, 28)))
        imresize(colorview(Gray, img), ratio=4)
    end

    arr = [do_img(i) for i=1:size(sms, 1)]
    imgs = mosaicview(arr, fillvalue=1, ncol=10, npad=4)
    save("samples.png", imgs) 
end

circuit, bpc, cutrain, cutest = run();
train(circuit, bpc, cutrain, cutest; num_epochs1 = 10);