using Test
using LogicCircuits
using ProbabilisticCircuits

function conditional_entropy_test()
    N = 10
    D = 5

    b = rand(Bool, N, D)
    dis_cache, hy_given_x = conditional_entropy(b, nothing; α=0)
    p = dis_cache.pairwise
    m = dis_cache.marginal

    for x in 1 : D, y in 1 : D
        h_y_x = p[x, y, 1] * log(p[x, y, 1] / m[x, 1]) +
                    p[x, y, 2] * log(p[x, y, 2] / m[x, 1]) +
                    p[x, y, 3] * log(p[x, y, 3] / m[x, 2]) +
                    p[x, y, 4] * log(p[x, y, 4] / m[x, 2])
        if x == y
            @test hy_given_x[x, y] == 0.0
        elseif !isnan(h_y_x)
            @test hy_given_x[x, y] ≈ - h_y_x atol=1e-12
        end
    end
end

@testset "Information Theory Test" begin
    conditional_entropy_test()
end

