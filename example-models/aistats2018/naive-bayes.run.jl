using Turing

include(Pkg.dir("Turing")*"/example-models/aistats2018/naive-bayes.data.jl")
include(Pkg.dir("Turing")*"/example-models/aistats2018/naive-bayes.model.jl")

turnprogress(false)

chn = sample(nb(data=nbmnistdata[1]), HMC(500, 0.05, 5))