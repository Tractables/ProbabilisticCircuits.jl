using Test
using .Juice

@testset "Circuit saver test" begin
    #@no_error begin
        # load and save as psdd
        circuit, vtree = load_struct_prob_circuit("circuits/little_4var.psdd", "circuits/little_4var.vtree");
        #TODO replace hardcoded temp file names by unique ones providd by Julia lib
        save_circuit("temp.psdd", circuit, vtree);
        save(vtree, "temp.vtree");
        circuit, vtree = load_struct_prob_circuit("temp.psdd", "temp.vtree");
        save_circuit("temp.psdd", circuit, vtree);

        # load and save as sdd
        circuit, vtree = load_struct_prob_circuit("circuits/little_4var.psdd", "circuits/little_4var.vtree");
        save_circuit("temp.sdd", circuit, vtree);
        save(vtree, "temp.vtree");

        rm("temp.sdd")
        rm("temp.vtree")
        rm("temp.psdd")
    #end
    #TODO add some actual @test statements
end
