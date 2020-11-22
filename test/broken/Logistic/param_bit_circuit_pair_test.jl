using Test
using LogicCircuits
using ProbabilisticCircuits
using Random

@testset "BitCircuitPair test" begin

end


@testset "ParamBitCircuitPair test" begin

    function test_integrity(pbc::ParamBitCircuitPair)
        bc = pbc.pair_bit
        @test num_elements(pbc) == size(bc.elements, 2)
        for el in 1:size(bc.elements,2)
            d = bc.elements[1,el]
            @test bc.nodes[1,d] <= el
            
            @test bc.nodes[2,d] >= el  #TODO this line fails
            p = bc.elements[2,el]
            @test el ∈ bc.parents[bc.nodes[3,p]:bc.nodes[4,p]]
            s = bc.elements[3,el]
            @test el ∈ bc.parents[bc.nodes[3,s]:bc.nodes[4,s]]
        end
        for node in 1:size(bc.nodes,2)
            first_el = bc.nodes[1,node]
            last_el = bc.nodes[2,node]
            if first_el != 0
                for i = first_el:last_el
                    @test bc.elements[1,i] == node #TODO this line fails (its node-1 instead for some reason)
                end
            end
            first_par = bc.nodes[3,node]
            last_par = bc.nodes[3,node]
            if first_par != 0
                for i = first_par:last_par
                    par = bc.parents[i]
                    @test bc.elements[2,par] == node || bc.elements[3,par] == node
                end
            else
                @test node == num_nodes(pbc) || node <= num_leafs(pbc)
            end
        end
        @test sum(length, bc.layers) == size(bc.nodes,2)
    end

    pc = zoo_psdd("exp-D15-N1000-C4.psdd");
    lc = zoo_lc("exp-D15-N1000-C4.circuit", 4);

    test_integrity(ParamBitCircuitPair(pc, children(lc)[1]));

end
