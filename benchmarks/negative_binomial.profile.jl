using Turing
using Distributions
using Base.Test

Profile.clear()

include("data.jl")
include("model.jl")

@profile sample(negbinmodel(negbindata), HMC(1000, 0.02, 1));

Profile.print(maxdepth = 13)
