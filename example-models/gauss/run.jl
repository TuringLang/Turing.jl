using Turing
using Distributions
using Base.Test

chain = sample(gaussmodel, gaussdata, SMC(300))
