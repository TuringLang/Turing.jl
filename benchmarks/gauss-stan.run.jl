using Distributions
using Turing
using Stan

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/simplegauss-stan.data.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/simplegauss-stan.model.jl")

stan_model_name = "simplegauss"
simplegaussstan = Stanmodel(name=stan_model_name, model=simplegaussstanmodel, nchains=1);

simple_gauss_stan_sim = stan(simplegaussstan, simplegaussstandata, CmdStanDir=CMDSTAN_HOME, summary=false);

s_stan = simple_gauss_stan_sim[1:1000, ["s"], :].value[:]
m_stan = simple_gauss_stan_sim[1:1000, ["m"], :].value[:]
sg_time = get_stan_time(stan_model_name)
