using LazyArtifacts
const zoo_version = "/Circuit-Model-Zoo-0.1.6"

zoo_psdd_file(name) = 
    artifact"circuit_model_zoo" * zoo_version * "/psdds/$name"
zoo_psdd(name) = 
    read(zoo_psdd_file(name), ProbCircuit, PsddFormat())
