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
