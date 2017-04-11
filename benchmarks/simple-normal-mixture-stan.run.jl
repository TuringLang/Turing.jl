using Mamba, Stan

include("simple-normal-mixture-stan.model.jl")
include("simple-normal-mixture-stan.data.jl")

simplenormalmixturestan = Stanmodel(name="normalmixture", model=simplenormalmixturemodel, nchains=1);

nm_stan_sim = stan(simplenormalmixturestan, simplenormalmixturestandata, CmdStanDir=CMDSTAN_HOME)
# describe(nm_stan_sim)

mu_1 = nm_stan_sim[1:1000, ["mu.1"], :].value[:]
mu_2 = nm_stan_sim[1:1000, ["mu.2"], :].value[:]
println("Result from Stan for Simple Normal Mixture model")
println("means from 1000 iterations:")
println("[$(mean(mu_1)),$(mean(mu_2))]")
