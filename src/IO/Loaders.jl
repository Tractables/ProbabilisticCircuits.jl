using LogicCircuits
using Pkg.Artifacts


#####################
# Circuit loaders
#####################

zoo_lc(name, num_classes) = 
    load_logistic_circuit(zoo_lc_file(name), num_classes)

zoo_clt_file(name) = 
    artifact"circuit_model_zoo" * "/Circuit-Model-Zoo-0.1.2/clts/$name"

zoo_clt(name) = 
    parse_clt(zoo_clt_file(name))

zoo_psdd_file(name) = 
    artifact"circuit_model_zoo" * "/Circuit-Model-Zoo-0.1.2/psdds/$name"

zoo_psdd(name) = 
    load_prob_circuit(zoo_psdd_file(name))
