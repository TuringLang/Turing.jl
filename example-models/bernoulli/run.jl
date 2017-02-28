using Turing
using Distributions
using Base.Test

include("data.jl")
include("model.jl")

sample(bermodel, berdata, HMC(1000, 0.1, 5))
