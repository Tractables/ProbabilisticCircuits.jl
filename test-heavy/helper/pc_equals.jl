using LogicCircuits: is⋁gate, is⋀gate

function test_pc_equals(c1, c2)
    @test num_nodes(c1) == num_nodes(c2)
    @test num_edges(c1) == num_edges(c2)
    for (n1, n2) in zip(linearize(c1), linearize(c2))
        if is⋁gate(n1)
            @test is⋁gate(n2)
            @test all(params(n1) ≈ params(n2))
        elseif is⋀gate(n1)
            @test is⋀gate(n2)
        else
            @test isliteralgate(n1) 
            @test isliteralgate(n2)
            @test literal(n1) == literal(n2)
        end
    end
end