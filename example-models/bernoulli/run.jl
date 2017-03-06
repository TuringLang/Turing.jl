using Turing
using Distributions
using Base.Test

include("data.jl")
include("model.jl")

chain = sample(bermodel, berdata, HMC(1000, 0.1, 5))

sim1 = TuringChains(chain)
describe(sim1)
