using DataFrames, ProbabilisticCircuits

include("load_mnist.jl")

# construct HCLT structure
train_df = DataFrame(train_bits, :auto);
num_hidden_cats = 32
@time circuit = hclt(size(train_bits,2); data = train_df, num_hidden_cats, num_trees = 1)
uniform_parameters!(circuit; perturbation = 0.4)

write("mnist_bits_hclt_$num_hidden_cats.jpc.gz", circuit)