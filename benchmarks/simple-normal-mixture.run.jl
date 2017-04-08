using Distributions
using Turing
using Base.Test

include("simple-normal-mixture.data.jl")
include("simple-normal-mixture.model.jl")

# NOTE: I only run a sub-set of the data as running the whole is quite slow
nmchain = sample(nmmodel(y[1:100]), Gibbs(250, PG(20, 1, :k), HMC(1, 0.2, 3, :mu, :theta)))
# describe(nmchain)
println("means:")
println(mean([[Float64(n) for n in ns] for ns in nmchain[:mu]]))
