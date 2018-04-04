using HDF5, JLD, ProfileView

using Turing
setadbackend(:reverse_diff)
turnprogress(false)

include(Pkg.dir("Turing")*"/example-models/stan-models/lda-stan.data.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/lda.model.jl")

sample(ldamodel(data=ldastandata[1]), HMC(2, 0.025, 10))
Profile.clear()
@profile sample(ldamodel(data=ldastandata[1]), HMC(2000, 0.025, 10))

ProfileView.svgwrite("ldamodel.svg")