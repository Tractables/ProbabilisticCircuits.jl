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

function little_3var_bernoulli(; p = 0.5)
    n1 = PlainInputNode(1, BernoulliDist(log(p)))
    n2 = PlainInputNode(2, BernoulliDist(log(p)))
    n3 = PlainInputNode(3, BernoulliDist(log(p)))
    summate(multiply(n1, n2, n3))
end

function little_3var_categorical(; num_cats = 3)
    n1 = PlainInputNode(1, CategoricalDist(num_cats))
    n2 = PlainInputNode(2, CategoricalDist(num_cats))
    n3 = PlainInputNode(3, CategoricalDist(num_cats))
    summate(multiply(n1, n2, n3))
end