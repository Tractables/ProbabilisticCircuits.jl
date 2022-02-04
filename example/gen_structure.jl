using DataFrames, ProbabilisticCircuits

include("load_mnist.jl")

# construct HCLT structure
train_gpu, test_gpu = mnist_gpu();
num_hidden_cats = 32;

@time circuit1 = hclt(train_gpu; latent_heuristic="vanila", num_cats=2, num_hidden_cats)
@time circuit2 = hclt(train_gpu; latent_heuristic="mixed", num_cats=2, num_hidden_cats)

println("Number of free parameters of vanila HCLT: $(num_parameters(circuit1))")
println("Number of free parameters of mixed HCLT: $(num_parameters(circuit2))")
println("Proportion of # parameters $(num_parameters(circuit2) / num_parameters(circuit1))")

uniform_parameters!(circuit; perturbation = 0.4)

write("mnist_bits_hclt_$num_hidden_cats.jpc.gz", circuit)