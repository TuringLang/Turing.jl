using Turing
using Distributions
using Base.Test

include("gauss.data.jl")
include("gauss.model.jl")

sim1 = @sample(gaussmodel(gaussdata), PG(20, 2000))
describe(sim1)

sim2 = @sample(gaussmodel(gaussdata), HMC(2000, 0.25, 5))
describe(sim2)

sim3 = @sample(gaussmodel(gaussdata), Gibbs(200, HMC(10, 0.25, 5, :mu), PG(20, 10, :lam)))
describe(sim3)
