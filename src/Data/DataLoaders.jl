#=
MyData:
- Julia version: 1.1.1
- Author: guy
- Date: 2019-06-25
=#
using MLDatasets
using CSV
using DataDeps

#####################
# Register data sources with DataDeps
#####################

expand_folds(f) = [f("train"), f("valid"), f("test")]

register(DataDep(
    "SampledBinaryMNIST",
    "Sampled Binary MNIST data",
    expand_folds(x -> "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_$x.amat"),
     "56e293409ebbdc08bcc7cbfe5453fbf7a9b86d0bb9b10d38a4919245566b7783"
))

const twenty_dataset_names = ["accidents", "ad", "baudio", "bbc", "bnetflix", "book", "c20ng", "cr52", "cwebkb",
           "dna", "jester", "kdd", "kosarek", "msnbc", "msweb", "nltcs", "plants", "pumsb_star", "tmovie", "tretail"]

register(DataDep(
    "20Datasets",
    "20 Density Estimation Datasets",
    flatmap(twenty_dataset_names) do ds
           expand_folds(x -> "https://raw.githubusercontent.com/UCLA-StarAI/Density-Estimation-Datasets/master/datasets/$ds/$ds.$x.data")
    end,
     #"38658a594750b17edcea50d82c0b7bde6c8298095f1d0ad1296a63b871c83377"
     "96a9fec15b4569aae8a0e5c4d92173b17b9e92ccde41d12cab15f270da851b9c"
))

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
    data_dir = datadep"20Datasets"
    function load(type)
        dataframe = CSV.read(data_dir*"/$name.$type.data"; header=false, delim=",",
                 truestrings=["1"], falsestrings=["0"], type=Bool, strict=true)
        XData(BitArray(Base.convert(Matrix{Bool}, dataframe)))
    end
    train = load("train")
    valid = load("valid")
    test = load("test")
    XDataset(train,valid,test)
end
