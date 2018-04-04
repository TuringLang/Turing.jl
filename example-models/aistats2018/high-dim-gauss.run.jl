using Turing

include(Pkg.dir("Turing")*"/example-models/aistats2018/high-dim-gauss.data.jl")
include(Pkg.dir("Turing")*"/example-models/aistats2018/high-dim-gauss.model.jl")

turnprogress(false)

mf = high_dim_gauss(data=hdgdata[1])
chn = sample(mf, HMC(1000, 0.05, 5))