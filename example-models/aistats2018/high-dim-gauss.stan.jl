include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
using Stan, HDF5, JLD

const hdgstanmodel = "
data {
  int D;
}
parameters {
  real mu[D];
}
model {
for (d in 1:D)
    mu[d] ~ normal(0, 1);
}
"

include(Pkg.dir("Turing")*"/example-models/aistats2018/high-dim-gauss.data.jl")

hdgstan = Stanmodel(Sample(algorithm=Stan.Hmc(Stan.Static(0.25),Stan.diag_e(),0.05,0.0), save_warmup=true,adapt=Stan.Adapt(engaged=false)), num_samples=1000, num_warmup=0, thin=1, name="High_Dim_Gauss", model=hdgstanmodel, nchains=1);

rc, hdg_stan_sim = stan(hdgstan, hdgdata, CmdStanDir=CMDSTAN_HOME, summary=false);

hdg_time = get_stan_time("High_Dim_Gauss")
println("Time used:", hdg_time)