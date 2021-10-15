using Revise
using LogicCircuits
using DataFrames
using ProbabilisticCircuits
using StatsBase: sample
using Random
using Serialization
using Images
using MLDatasets

function mnist(labeled = false)
    # transposing makes slicing by variable much much faster
    # need to take a copy to physically move the data around
    train_x = collect(Float32, transpose(reshape(MNIST.traintensor(), 28*28, :)))
    test_x  = collect(Float32, transpose(reshape(MNIST.testtensor(), 28*28, :)))
    
    train = DataFrame(train_x, :auto)
    valid = nothing # why is there no validation set in `MLDataSets`??
    test = DataFrame(test_x, :auto)
    if (labeled)
        train_y::Vector{UInt8} = MNIST.trainlabels()
        test_y::Vector{UInt8}  = MNIST.testlabels()
        train.y = train_y
        test.y = test_y
    end
    return train, valid, test
end

function arr2img(one_sample; ratio = 1)
    imresize(colorview(Gray, transpose(reshape(Array(one_sample), (28,28)))), ratio=ratio)
end

import LogicCircuits: example
function example(data::Vector{DataFrame}, i)
    for b = 1 : size(data)[1]
        if i <  size(data[b])[1]
            return data[b][i,:]
        end
        i -= size(data[b])[1]
    end
end

function load_pc(name)
    read("$(@__DIR__)//$(name).jpc", ProbCircuit)
end
 
function main()

    Random.seed!(2880)
    pc_name = "mnist_cats8_181"  # "mnist_b_301"
    img_idx = 10
    soft_reg_train = 0.002
    batch_size = 1024

    pc = load_pc(pc_name)
    nvars = num_variables(pc)

    println(num_nodes(pc))
    println(num_variables(pc))
    println(num_parameters(pc))

    t = @elapsed begin
        # MNIST dataset
        mnist_train, _, mnist_test = mnist(true);
        mnist_train, _, mnist_test = threshold(mnist_train[:, 1:end-1], nothing, mnist_test[:, 1:end-1], 0.5);
        # mnist_train = shuffle_examples(mnist_train);
        mnist_valid = mnist_train[1:10000, :];
        mnist_train = mnist_train[10001:end, :];
        
        train_data = batch(soften(mnist_train, soft_reg_train; scale_by_marginal = true), batch_size; shuffle=false)
        valid_data = batch(mnist_valid, batch_size; shuffle=false)
        test_data = batch(mnist_test, batch_size; shuffle=false)
    end

    
    img = example(test_data, img_idx)
    save("mnist_image_$(img_idx).png", arr2img(img))

    # choose the conditoin index
    # 1. condition on pixels 1..K,
    # 2. condition on indexes nvars...(nvars-B)
    println("Compute pixels idxs to condition on")
    # K = 28*12
    # B = 28*12
    # condition_indexs = [i for i=1:K]
    # append!(condition_indexs, [i for i=nvars-1:-1:(nvars-B+1)])
    # partial_image = [img[i] == 1 ? i : -i for i in condition_indexs];

    U = 14
    D = 7
    L = 16
    R = 4

    condition_indexs = [i for i=1:nvars if ((i%28 < L) | (i % 28 > 28-R) | (i/28 < U) | (i/28 > 28-D))]
    partial_image = [img[i] == 1 ? i : -i for i in condition_indexs];

    println("Compute query variables (rest of pixels)")
    # query to be the rest of pixels
    query = []
    for i = 1:nvars
        if !(i in condition_indexs)
            push!(query, i)
        end
    end
    query = BitSet(query);

    # Gray out is missing, the rest is the condition
    println("Save evidence image to file....png")
    img_temp = Float32.(Array(example(test_data, img_idx)))
    img_temp_2 = Float32.(Array(example(test_data, img_idx)))
    for i in 1:nvars
        if !(i in condition_indexs)
            img_temp[i] = 0.5
        else
            img_temp_2[i] = 0.5
        end
    end
    save("mnist_evidence_$(img_idx).png",  arr2img(img_temp))

    # Black pixels are the query
    println("Save query image to file....png")
    query_img = [i in query ? 0.0 : 1.0 for i=1:nvars]
    save("mnist_query_$(img_idx).png",  arr2img(query_img))

    t = @elapsed begin
        println(num_nodes(pc), " ", num_parameters(pc), " ", num_edges(pc))
        conjoin_lits = Int32.(partial_image)
        println("conjoin_with_literals .. $(size(conjoin_lits))")
        pc2 = conjoin_with_literals(pc, conjoin_lits) 
        println(num_nodes(pc2), " ", num_parameters(pc2), " ", num_edges(pc2))
    end

    # chunk = 20
    # conjoin_lits = Int32.(partial_image)
    # for i = 1:chunk:size(conjoin_lits)[1]
    #     end_idx = min(size(conjoin_lits)[1], i+chunk-1)
    #     print("Conjoining on conjoin_lits[$(i):$(end_idx)] / $(size(conjoin_lits)[1])")
    #     t = @elapsed begin
    #         pc = pc_condition(pc, conjoin_lits[i:end_idx]...)    
    #     end
    #     print(" $(t) seconds;")
    #     println(";; Nodes $(num_nodes(pc)); Params $(num_parameters(pc));")
    #     write("$(pc_name)_conjoined.jpc", pc)
    # end

    arr = [arr2img(img), arr2img(img_temp), arr2img(img_temp_2),  arr2img(query_img)]
    save("$(@__DIR__)//mnist_$(img_idx)_before_map.png", mosaicview(arr, fillvalue=1, npad=4, ncol=10, rowmajor=true))
    did_timeout, total_time, iter, ub, lb, lb_state, pc  = mmap_solve(pc2, query, heur="UB", verbose=true);
   
    for i in condition_indexs
        lb_state[i] = img[i]
    end
    lb_state = Array{Bool}(lb_state);

    #save("mnist_final.png", arr2img(lb_state))
    arr = [arr2img(img), arr2img(img_temp), arr2img(img_temp_2), arr2img(query_img), arr2img(lb_state)]
    save("$(@__DIR__)//mnist_$(img_idx)_map_final_$(pc_name).png", mosaicview(arr, fillvalue=1, npad=4, ncol=10, rowmajor=true))
end

main()