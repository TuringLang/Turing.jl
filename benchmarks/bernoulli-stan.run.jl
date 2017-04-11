using Mamba, Stan

include("bernoulli-stan.data.jl")
include("bernoulli-stan.model.jl")

berstan = Stanmodel(name="bernoulli", model=berstanmodel, nchains=1);

ber_stan_sim = stan(berstan, berstandata, CmdStanDir=CMDSTAN_HOME)
describe(ber_stan_sim)

theta_stan = ber_stan_sim[1:1000, ["theta"], :].value[:]
theta_turing = ber_sim[:theta]

println("The difference between the means from Stan and Turing is:")
println("  E(theta_stan) - E(theta_turing) = $(mean(theta_stan) - mean(theta_turing))")
