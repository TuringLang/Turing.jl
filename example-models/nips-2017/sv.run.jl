using Distributions
using Turing

TPATH = Pkg.dir("Turing")

include("/example-models/nips-2017/sv.model.jl")

PR_for_distributions_is_merged = false

if PR_for_distributions_is_merged

using HDF5, JLD
sv_data = load(TPATH*"/example-models/nips-2017/sv-data.jld")["data"]

model_f = sv_model(data=sv_data[1])
sample_n = 10000

setchunksize(550)
chain_nuts = sample(model_f, NUTS(sample_n, 0.65))
decribe(chain_nuts)

setchunksize(5)
chain_gibbs = sample(model_f, Gibbs(sample_n, PG(50,1,:h),NUTS(1000,0.65,:ϕ,:σ,:μ)))
decribe(chain_gibbs)

end
