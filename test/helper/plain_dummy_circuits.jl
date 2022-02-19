using ProbabilisticCircuits
using ProbabilisticCircuits: PlainInputNode

function little_2var()
    pos = PlainInputNode(1, Literal(true))
    neg = PlainInputNode(1, Literal(false))
    sum1 = pos + neg
    sum2 = pos + neg
    pos = PlainInputNode(2, Literal(true))
    neg = PlainInputNode(2, Literal(false))
    mul1 = pos * sum1
    mul2 = neg * sum2
    mul1 + mul2
end

function little_3var()
    sum1 = little_2var()
    pos = PlainInputNode(3, Literal(true))
    neg = PlainInputNode(3, Literal(false))
    sum2 = summate(inputs(sum1))
    mul1 = pos * sum1
    mul2 = neg * sum2
    mul1 + mul2
end

function little_3var_bernoulli(; p = 0.5)
    n1 = PlainInputNode(1, Bernoulli(log(p)))
    n2 = PlainInputNode(2, Bernoulli(log(p)))
    n3 = PlainInputNode(3, Bernoulli(log(p)))
    summate(multiply(n1, n2, n3))
end

function little_3var_categorical(; num_cats = 3)
    n1 = PlainInputNode(1, Categorical(num_cats))
    n2 = PlainInputNode(2, Categorical(num_cats))
    n3 = PlainInputNode(3, Categorical(num_cats))
    summate(multiply(n1, n2, n3))
end

function little_4var()
    circuit = IOBuffer(b"""psdd 19
    L 0 0 1 
    L 2 1 2
    L 4 2 3
    L 6 3 4
    L 1 0 -1
    L 3 1 -2
    L 5 2 -3
    L 7 3 -4
    D 9 5 4 4 6 -1.6094379124341003 4 7 -1.2039728043259361 5 6 -0.916290731874155 5 7 -2.3025850929940455
    D 8 4 4 0 2 -2.3025850929940455 0 3 -2.3025850929940455 1 2 -2.3025850929940455 1 3 -0.35667494393873245
    D 10 6 1 8 9 0.0
    """)

    prob_circuit = read(circuit, ProbCircuit, ProbabilisticCircuits.PsddFormat());
end


"""
Generates all possible binary configurations of size N
"""
function generate_data_all(N::Int)
    data_all = transpose(parse.(Bool, split(bitstring(0)[end-N+1:end], "")));
    for mask = 1: (1<<N) - 1
        data_all = vcat(data_all,
            transpose(parse.(Bool, split(bitstring(mask)[end-N+1:end], "")))
        );
    end
    Matrix(data_all)
end