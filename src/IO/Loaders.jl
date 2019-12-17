using ..Data
using MLDatasets
using CSV
using Pkg.Artifacts

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

expand_folds(f) = [f("train"), f("valid"), f("test")]

const twenty_dataset_names = ["accidents", "ad", "baudio", "bbc", "bnetflix", "book", "c20ng", "cr52", "cwebkb",
           "dna", "jester", "kdd", "kosarek", "msnbc", "msweb", "nltcs", "plants", "pumsb_star", "tmovie", "tretail", 
           "binarized_mnist"]


sampled_mnist() = twenty_datasets("binarized_mnist")

function twenty_datasets(name)
    @assert in(name, twenty_dataset_names)
    data_dir = artifact"density_estimation_datasets"
    function load(type)
        dataframe = CSV.read(data_dir*"/Density-Estimation-Datasets-1.0.1/datasets/$name/$name.$type.data"; header=false,
                 truestrings=["1"], falsestrings=["0"], type=Bool, strict=true)
        XData(BitArray(Base.convert(Matrix{Bool}, dataframe)))
    end
    train = load("train")
    valid = load("valid")
    test = load("test")
    XDataset(train,valid,test)
end

#####################
# Circuit loaders
#####################

zoo_vtree_file(name) = 
    artifact"circuit_model_zoo" * "/Circuit-Model-Zoo-0.1.1/vtrees/$name"

zoo_vtree(name) = 
    load_vtree(zoo_vtree_file(name))

zoo_cnf_file(name) = 
    artifact"circuit_model_zoo" * "/Circuit-Model-Zoo-0.1.1/cnfs/$name"

zoo_cnf(name) = 
    load_cnf(zoo_cnf_file(name))

zoo_dnf_file(name) = 
    artifact"circuit_model_zoo" * "/Circuit-Model-Zoo-0.1.1/dnfs/$name"

zoo_dnf(name) = 
    load_dnf(zoo_dnf_file(name))

zoo_lc_file(name) = 
    artifact"circuit_model_zoo" * "/Circuit-Model-Zoo-0.1.1/lcs/$name"

zoo_clt_file(name) = 
    artifact"circuit_model_zoo" * "/Circuit-Model-Zoo-0.1.1/clt/$name"

zoo_psdd_file(name) = 
    artifact"circuit_model_zoo" * "/Circuit-Model-Zoo-0.1.1/psdds/$name"

zoo_sdd_file(name) = 
    artifact"circuit_model_zoo" * "/Circuit-Model-Zoo-0.1.1/sdds/$name"