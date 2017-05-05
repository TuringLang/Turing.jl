using Distributions
using Turing
using Stan

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/bernoulli-stan.data.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/bernoulli-stan..model.jl")

stan_model_name = "bernoulli"
berstan = Stanmodel(name=stan_model_name, model=berstanmodel, nchains=1);

ber_stan_sim = stan(berstan, berstandata, CmdStanDir=CMDSTAN_HOME, summary=false)

theta_stan = ber_stan_sim[1:1000, ["theta"], :].value[:]
ber_time = get_stan_time(stan_model_name)
