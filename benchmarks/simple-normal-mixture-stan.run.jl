include("simple-normal-mixture-stan.model.jl")
include("simple-normal-mixture-stan.data.jl")

stan_model_name = "normalmixture"
simplenormalmixturestan = Stanmodel(name=stan_model_name, model=simplenormalmixturemodel, nchains=1);

nm_stan_sim = stan(simplenormalmixturestan, [Dict("N"=>100, "y"=>simplenormalmixturestandata[1]["y"][1:100])], CmdStanDir=CMDSTAN_HOME, summary=false)
# describe(nm_stan_sim)

nm_theta = nm_stan_sim[1:1000, ["theta"], :].value[:]
nm_mu_1 = nm_stan_sim[1:1000, ["mu.1"], :].value[:]
nm_mu_2 = nm_stan_sim[1:1000, ["mu.2"], :].value[:]
nm_time = get_stan_time(stan_model_name)
