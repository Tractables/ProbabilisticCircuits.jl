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

function little_3var_bernoulli(firstvar=1; p = 0.5)
    n1 = PlainInputNode(firstvar, Bernoulli(log(p)))
    n2 = PlainInputNode(firstvar+1, Bernoulli(log(p)))
    n3 = PlainInputNode(firstvar+2, Bernoulli(log(p)))
    summate(multiply(n1, n2, n3))
end

function little_3var_categorical(firstvar=1; num_cats = 3)
    n1 = PlainInputNode(firstvar, Categorical(num_cats))
    n2 = PlainInputNode(firstvar+1, Categorical(num_cats))
    n3 = PlainInputNode(firstvar+2, Categorical(num_cats))
    summate(multiply(n1, n2, n3))
end

function little_3var_binomial(firstvar=1; n = 10)
    n1 = PlainInputNode(firstvar, Binomial(n, 0.1))
    n2 = PlainInputNode(firstvar+1, Binomial(n, 0.5))
    n3 = PlainInputNode(firstvar+2, Binomial(n, 0.9))
    summate(multiply(n1, n2, n3))
end

function little_gmm(firstvar=1; sigma = 1)
    n1 = PlainInputNode(firstvar, Gaussian(-1.0, sigma))
    n2 = PlainInputNode(firstvar, Gaussian(1.0, sigma))
    
    0.5 * n1 + 0.5 * n2 
end

function little_2var_gmm(firstvar=1; sigma = 1)
    n1_x = PlainInputNode(firstvar, Gaussian(-2.0, sigma))
    n1_y = PlainInputNode(firstvar+1, Gaussian(-2.0, sigma))

    n2_x = PlainInputNode(firstvar, Gaussian(0.0, sigma))
    n2_y = PlainInputNode(firstvar+1, Gaussian(0.0, sigma))

    n1 = multiply(n1_x, n1_y)
    n2 = multiply(n2_x, n2_y)

    0.2 * n1 + 0.8 * n2
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

function little_2var_indicator(firstvar=0)
    v1 = PlainInputNode(firstvar, Indicator(0))
    v2 = PlainInputNode(firstvar, Indicator(1))
    v3 = PlainInputNode(firstvar, Indicator(2))
    sum1 = v1 + v2 + v3
    sum2 = v1 + v2 + v3
    sum3 = v1 + v2 + v3
    v1 = PlainInputNode(firstvar+1, Indicator(0))
    v2 = PlainInputNode(firstvar+1, Indicator(1))
    v3 = PlainInputNode(firstvar+1, Indicator(2))
    mul1 = v1 * sum1
    mul2 = v2 * sum2
    mul3 = v3 * sum3
    mul1 + mul2 + mul3
end

function little_hybrid_circuit()
    x = little_3var()
    y = little_2var_indicator(4)
    z1 = little_3var_bernoulli(7)
    z2 = little_3var_categorical(7)
    (x * y * z1) + (x * y * z2) 
end