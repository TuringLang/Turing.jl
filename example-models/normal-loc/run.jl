using Distributions
using Turing
using Base.Test

include("data.jl")
include("model.jl")

nlchain = sample(nlmodel, nldata, HMC(1000, 0.05, 3))
print(nlchain[:mu])
