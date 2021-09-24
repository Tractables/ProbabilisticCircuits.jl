using Test
using LogicCircuits
using ProbabilisticCircuits

function test_equal(c1::LogicCircuit, c2::LogicCircuit)
    result = true
    for (n1, n2) in zip(linearize(c1), linearize(c2))
        if is⋁gate(n1)
            @test is⋁gate(n2)
            @test all(params(n1) ≈ params(n2))
        elseif is⋀gate(n1)
            result = result
            @test is⋀gate(n2)
        else
            @test isliteralgate(n1) 
            @test isliteralgate(n2)
            @test literal(n1) == literal(n2)
        end
    end
    result
end

@testset "Circuit saver test" begin
    mktempdir() do tmp

        circuit, vtree = load_struct_prob_circuit(
                            zoo_psdd_file("little_4var.psdd"), zoo_vtree_file("little_4var.vtree"))

        # load, save, and load as .psdd
        @test_nowarn save_circuit("$tmp/temp.psdd", circuit, vtree)
        @test_nowarn save_vtree("$tmp/temp.vtree", vtree);

        circuit2, vtree2 = load_struct_prob_circuit("$tmp/temp.psdd", "$tmp/temp.vtree")
        test_equal(circuit, circuit2)

        f_c, f_v = open("$tmp/temp.psdd", "r"), open("$tmp/temp.vtree", "r")
        circuit2, _ = load_struct_prob_circuit(f_c, f_v)
        test_equal(circuit, circuit2)
        close(f_c); close(f_v)

        f = open("$tmp/temp.psdd", "w")
        @test_nowarn save_as_psdd(f, circuit, vtree)
        close(f)
        circuit2, _ = load_struct_prob_circuit("$tmp/temp.psdd", "$tmp/temp.vtree")
        test_equal(circuit, circuit2)

        # save and load as .sdd
        @test_nowarn save_circuit("$tmp/temp.sdd", PlainStructLogicCircuit(circuit), vtree)
        @test_nowarn save_vtree("$tmp/temp.vtree", vtree)

        @test_nowarn save_as_tex("$tmp/temp.tex", circuit)
        @test_nowarn save_as_dot("$tmp/temp.tex", circuit)
        @test_nowarn plot(circuit)
        @test_nowarn plot(vtree)

        # psdd2
        v = Vtree(5, :balanced)
        c = fully_factorized_circuit(ProbCircuit, v).children[1]
        @test_nowarn save_circuit("$tmp/temp.psdd", c, v)
        @test_nowarn save_vtree("$tmp/temp.vtree", v);
        c2, v2 = load_struct_prob_circuit("$tmp/temp.psdd", "$tmp/temp.vtree")
        test_equal(c, c2)
    end

    mktempdir() do tmp
        classes = 2
        lc1 = zoo_lc("little_4var.circuit", classes)
        @test_nowarn save_circuit("$tmp/temp.circuit", lc1, nothing)
        
        lc2 = load_logistic_circuit("$tmp/temp.circuit", classes)
        test_equal(lc1, lc2)
    end
end
