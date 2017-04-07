using Turing
using Distributions
using Base.Test

include("data.jl")
include("model.jl")

# Produce an error.
sample(negbinmodel(negbindata), HMC(1000, 0.02, 1));
