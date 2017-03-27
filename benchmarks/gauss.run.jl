using Turing
using Distributions
using Base.Test

include("gauss.data.jl")
include("gauss.model.jl")

sim1 = @sample(gaussmodel(gaussdata), PG(20, 2000))
describe(sim1)

sim2 = @sample(gaussmodel(gaussdata), HMC(2000, 0.25, 5))
describe(sim2)
