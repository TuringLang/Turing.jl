using Distributions
using Turing
using Base.Test

include("data.jl")
include("model.jl")

nmchain = sample(nmmodel, nmdata, HMC(1000, 0.05, 3))
print(mean([[Float64(n) for n in ns] for ns in nmchain[:mu]]))
