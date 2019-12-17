using Test
using Juice

@testset "Circuit saver test" begin
    mktempdir() do tmp
        # load and save as psdd
        circuit, vtree = load_struct_prob_circuit(
                            zoo_psdd_file("little_4var.psdd"), zoo_vtree_file("little_4var.vtree"))
        #TODO replace hardcoded temp file names by unique ones providd by Julia lib
        save_circuit(tmp*"/temp.psdd", circuit, vtree)
        save(vtree, tmp*"/temp.vtree");
        circuit, vtree = load_struct_prob_circuit(tmp*"/temp.psdd", tmp*"/temp.vtree")
        save_circuit(tmp*"/temp.psdd", circuit, vtree)

        # load and save as sdd
        circuit, vtree = load_struct_prob_circuit(
                            zoo_psdd_file("little_4var.psdd"), zoo_vtree_file("little_4var.vtree"))
        save_circuit(tmp*"/temp.sdd", circuit, vtree)
        save(vtree, tmp*"/temp.vtree")
    end
    
    #TODO add some actual @test statements
    @test true
end
