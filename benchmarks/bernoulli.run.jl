using Turing
using Distributions
using Base.Test

include("bernoulli.data.jl")
include("bernoulli.model.jl")

ber_sim = sample(bermodel(berdata), Turing.HMC(1000, 0.25, 5))

describe(ber_sim)
