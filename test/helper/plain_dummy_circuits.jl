using ProbabilisticCircuits
using ProbabilisticCircuits: PlainInputNode

function little_2var()
    pos = PlainInputNode(1, LiteralDist(true))
    neg = PlainInputNode(1, LiteralDist(false))
    sum1 = pos + neg
    sum2 = pos + neg
    pos = PlainInputNode(2, LiteralDist(true))
    neg = PlainInputNode(2, LiteralDist(false))
    mul1 = pos * sum1
    mul2 = neg * sum2
    mul1 + mul2
end

function little_3var()
    sum1 = little_2var()
    pos = PlainInputNode(3, LiteralDist(true))
    neg = PlainInputNode(3, LiteralDist(false))
    sum2 = summate(inputs(sum1))
    mul1 = pos * sum1
    mul2 = neg * sum2
    mul1 + mul2
end

function little_3var_bernoulli(; p::Float32 = Float32(0.5))
    n1 = input_node(ProbCircuit, BernoulliDist, 1; p)
    n2 = input_node(ProbCircuit, BernoulliDist, 2; p)
    n3 = input_node(ProbCircuit, BernoulliDist, 3; p)
    summate(multiply(n1, n2, n3))
end

function little_3var_categorical(; num_cats::UInt32 = UInt32(3))
    n1 = input_node(ProbCircuit, CategoricalDist, 1; num_cats)
    n2 = input_node(ProbCircuit, CategoricalDist, 2; num_cats)
    n3 = input_node(ProbCircuit, CategoricalDist, 3; num_cats)
    summate(multiply(n1, n2, n3))
end