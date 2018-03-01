using Turing

include(Pkg.dir("Turing")*"/example-models/aistats2018/high-dim-gauss.data.jl")
include(Pkg.dir("Turing")*"/example-models/aistats2018/high-dim-gauss.model.jl")

chn = sample(high_dim_gauss(data=hdgdata[1]), HMC(500, 0.05, 5))