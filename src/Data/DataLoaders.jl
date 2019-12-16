#=
MyData:
- Julia version: 1.1.1
- Author: guy
- Date: 2019-06-25
=#
using MLDatasets
using CSV
using DataDeps
using Pkg.Artifacts

#####################
# Register data sources with DataDeps
#####################

function __init__()
    register(DataDep(
        "SampledBinaryMNIST",
        "Sampled Binary MNIST data",
        expand_folds(x -> "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_$x.amat"),
         "56e293409ebbdc08bcc7cbfe5453fbf7a9b86d0bb9b10d38a4919245566b7783"
    ))
end

expand_folds(f) = [f("train"), f("valid"), f("test")]

const twenty_dataset_names = ["accidents", "ad", "baudio", "bbc", "bnetflix", "book", "c20ng", "cr52", "cwebkb",
           "dna", "jester", "kdd", "kosarek", "msnbc", "msweb", "nltcs", "plants", "pumsb_star", "tmovie", "tretail"]


#####################
# Data loaders
#####################

function dataset(data; do_threshold=false, do_shuffle=false, batch_size=-1)
    shuffled_data = do_shuffle ? shuffle(data) : data
    thresholded_data = do_threshold ? threshold(shuffled_data) : shuffled_data
    batch_size > 0 ? batch(thresholded_data, batch_size) : thresholded_data
end

function mnist()
    # transposing makes slicing by variable much much faster
    # need to take a copy to physically move the data around
    train_x = copy_with_eltype(transpose(MNIST.convert2features(MNIST.traintensor())), Float32)
    test_x  = copy_with_eltype(transpose(MNIST.convert2features(MNIST.testtensor())), Float32)

    train_y::Vector{UInt8} = MNIST.trainlabels()
    test_y::Vector{UInt8}  = MNIST.testlabels()

    train = XYData(XData(train_x),train_y)
    valid = nothing
    test = XYData(XData(test_x),test_y)

    XYDataset(train,valid,test)
end

function sampled_mnist()
    data_dir = datadep"SampledBinaryMNIST"
    function load(type)
        dataframe = CSV.read(data_dir*"/binarized_mnist_$type.amat"; header=false, delim=" ",
                 truestrings=["1"], falsestrings=["0"], type=Bool, strict=true)
        XData(BitArray(Base.convert(Matrix{Bool}, dataframe)))
    end
    train = load("train")
    valid = load("valid")
    test = load("test")
    XDataset(train,valid,test)
end

function twenty_datasets(name)
    @assert in(name, twenty_dataset_names)
    data_dir = artifact"twenty_datasets"
    function load(type)
        dataframe = CSV.read(data_dir*"/Density-Estimation-Datasets-1.0/datasets/$name/$name.$type.data"; header=false, delim=",",
                 truestrings=["1"], falsestrings=["0"], type=Bool, strict=true)
        XData(BitArray(Base.convert(Matrix{Bool}, dataframe)))
    end
    train = load("train")
    valid = load("valid")
    test = load("test")
    XDataset(train,valid,test)
end
