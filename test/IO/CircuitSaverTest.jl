using Test
using LogicCircuits
using ProbabilisticCircuits

@testset "Circuit saver test" begin
    mktempdir() do tmp

        circuit, vtree = load_struct_prob_circuit(
                            zoo_psdd_file("little_4var.psdd"), zoo_vtree_file("little_4var.vtree"))

        # load, save, and load as .psdd
        save_circuit("$tmp/temp.psdd", circuit, vtree)
        save(vtree, "$tmp/temp.vtree");
        load_struct_prob_circuit("$tmp/temp.psdd", "$tmp/temp.vtree")
        
        # save and load as .sdd
        save_circuit("$tmp/temp.sdd", circuit, vtree)
        save(vtree, "$tmp/temp.vtree")

    end
    
    #TODO add some actual @test statements
    @test true
end
