using Distributions, Turing, Stan, Mamba

include("benchmarkhelper.jl")
include("lda-stan.data.jl")
include("lda-stan.model.jl")

stan_model_name = "LDA"
ldastan = Stanmodel(Sample(save_warmup=true), name=stan_model_name, model=ldastanmodel, nchains=1);

lda_stan_sim = stan(ldastan, ldastandata, CmdStanDir=CMDSTAN_HOME, summary=false);
# lda_stan_sim.names

lda_stan_d_raw = Dict()
for i = 1:2, j = 1:5
  lda_stan_d_raw["phi[$i][$j]"] = lda_stan_sim[1001:2000, ["phi.$i.$j"], :].value[:]
end

lda_stan_d = Dict()
for i = 1:2
  lda_stan_d["phi[$i]"] = mean([[lda_stan_d_raw["phi[$i][$k]"][j] for k = 1:5] for j = 1:1000])
end

lda_time = get_stan_time(stan_model_name)
