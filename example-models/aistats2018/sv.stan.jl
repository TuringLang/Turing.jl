# https://github.com/stan-dev/example-models/blob/master/misc/moving-avg/stochastic-volatility.stan
include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
using Stan, HDF5, JLD

const svstanmodel = "
data {
  int<lower=0> T;   // # time points (equally spaced)
  vector[T] y;      // mean corrected return at time t
}
parameters {
  real mu;                     // mean log volatility
  real<lower=-1,upper=1> phi;  // persistence of volatility
  real<lower=0> sigma;         // white noise shock scale
  vector[T] h;                 // log volatility at time t
}
model {
  phi ~ uniform(-1,1);
  sigma ~ cauchy(0,5);
  mu ~ cauchy(0,10);  
  h[1] ~ normal(mu, sigma / sqrt(1 - phi * phi));
  for (t in 2:T)
    h[t] ~ normal(mu + phi * (h[t - 1] -  mu), sigma);
  for (t in 1:T)
    y[t] ~ normal(0, exp(h[t] / 2));
}
"

sv_data = load(Pkg.dir("Turing")*"/example-models/nips-2017/sv-data.jld.data")["data"]

svstan = Stanmodel(Sample(algorithm=Stan.Hmc(Stan.Static(0.5),Stan.diag_e(),0.05,0.0), save_warmup=true,adapt=Stan.Adapt(engaged=false)), num_samples=2000, num_warmup=0, thin=1, name="Stochastic_Volatility", model=svstanmodel, nchains=1);

rc, sv_stan_sim = stan(svstan, sv_data, CmdStanDir=CMDSTAN_HOME, summary=false);

sv_time = get_stan_time("Stochastic_Volatility")
println("Time used:", sv_time)