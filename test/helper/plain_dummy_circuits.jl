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