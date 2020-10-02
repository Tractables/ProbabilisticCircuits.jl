using Test
using LogicCircuits
using ProbabilisticCircuits

@testset "ParamBitCircuit test" begin

    function test_integrity(pbc)
        bc = pbc.bitcircuit
        params = pbc.params
        @test num_elements(pbc) == size(bc.elements, 2)
        for el in 1:size(bc.elements,2)
            d = bc.elements[1,el]
            @test bc.nodes[1,d] <= el
            @test bc.nodes[2,d] >= el
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
                    @test bc.elements[1,i] == node
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

    r = fully_factorized_circuit(PlainProbCircuit, 10)
    test_integrity(ParamBitCircuit(r, 10))

    r = fully_factorized_circuit(LogisticCircuit, 10; classes=2)
    test_integrity(ParamBitCircuit(r, 2, 10))
    
    vtree = PlainVtree(10, :balanced)
    r = fully_factorized_circuit(StructProbCircuit, vtree)
    test_integrity(ParamBitCircuit(r, 10))


end
