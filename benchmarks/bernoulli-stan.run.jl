include("bernoulli-stan.data.jl")
include("bernoulli-stan.model.jl")

berstan = Stanmodel(name="bernoulli", model=berstanmodel, nchains=1);

ber_stan_sim = stan(berstan, berstandata, CmdStanDir=CMDSTAN_HOME, summary=false)

theta_stan = ber_stan_sim[1:1000, ["theta"], :].value[:]
