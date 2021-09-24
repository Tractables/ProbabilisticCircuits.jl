using Test
using LogicCircuits
using ProbabilisticCircuits



        # psdd2
        v = Vtree(5, :balanced)
        c = fully_factorized_circuit(ProbCircuit, v).children[1]
        @test_nowarn save_circuit("$tmp/temp.psdd", c, v)
        @test_nowarn save_vtree("$tmp/temp.vtree", v);
        c2, v2 = load_struct_prob_circuit("$tmp/temp.psdd", "$tmp/temp.vtree")
        test_equal(c, c2)
end
