using Distributions
using Turing
using Stan

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/MoC-stan.data.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/MoC-stan.model.jl")

stan_model_name = "Naive_Bayes"
nbstan = Stanmodel(Sample(algorithm=Stan.Hmc(Stan.Static(0.05),Stan.diag_e(),0.01,0.0),
  save_warmup=true,adapt=Stan.Adapt(engaged=false)),
  num_samples=2000, num_warmup=0, thin=1,
name=stan_model_name, model=naivebayesstanmodel, nchains=1);

rc, nb_stan_sim = stan(nbstan, nbstandata, CmdStanDir=CMDSTAN_HOME, summary=false);
# nb_stan_sim.names

stan_d_raw = Dict()
for i = 1:4, j = 1:10
  stan_d_raw["phi[$i][$j]"] = nb_stan_sim[1001:2000, ["phi.$i.$j"], :].value[:]
end

stan_d = Dict()
for i = 1:4
  stan_d["phi[$i]"] = mean([[stan_d_raw["phi[$i][$k]"][j] for k = 1:10] for j = 1:1000])
end

nb_time = get_stan_time(stan_model_name)
