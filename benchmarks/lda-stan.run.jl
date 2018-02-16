using Distributions
using Turing
using Stan

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/lda-stan.data.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/lda-stan.model.jl")

stan_model_name = "LDA"
# ldastan = Stanmodel(Sample(save_warmup=true), name=stan_model_name, model=ldastanmodel, nchains=1);
# To understand parameters, use: ?Stan.Static, ?Stan,Hmc
ldastan = Stanmodel(Sample(algorithm=Stan.Hmc(Stan.Static(0.25),Stan.diag_e(),0.025,0.0),
  save_warmup=true,adapt=Stan.Adapt(engaged=false)),
  num_samples=2000, num_warmup=0, thin=1,
  name=stan_model_name, model=ldastanmodel, nchains=1);

rc, lda_stan_sim = stan(ldastan, ldastandata, CmdStanDir=CMDSTAN_HOME, summary=false);
# lda_stan_sim.names
V = ldastandata[1]["V"]
K = ldastandata[1]["K"]
lda_stan_d_raw = Dict()
for i = 1:K, j = 1:V
  lda_stan_d_raw["phi[$i][$j]"] = lda_stan_sim[1001:2000, ["phi.$i.$j"], :].value[:]
end

lda_stan_d = Dict()
for i = 1:K
  lda_stan_d["phi[$i]"] = mean([[lda_stan_d_raw["phi[$i][$k]"][j] for k = 1:V] for j = 1:1000])
end

lda_time = get_stan_time(stan_model_name)
