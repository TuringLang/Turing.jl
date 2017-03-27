using Turing
using Distributions
using Base.Test

include("bernoulli.data.jl")
include("bernoulli.model.jl")

sim1 = @sample(bermodel(berdata), HMC(2000, 0.25, 5))

describe(sim1)
