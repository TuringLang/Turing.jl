include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")

using Stan, Mamba

# Model taken from https://github.com/goedman/Stan.jl/blob/master/Examples/Mamba/EightSchools/schools8.jl

const eightschools ="
data {
  int<lower=0> J; // number of schools
  real y[J]; // estimated treatment effects
  real<lower=0> sigma[J]; // s.e. of effect estimates
}
parameters {
  real mu;
  real<lower=0> tau;
  real eta[J];
}
transformed parameters {
  real theta[J];
  for (j in 1:J)
    theta[j] <- mu + tau * eta[j];
}
model {
  eta ~ normal(0, 1);
  y ~ normal(theta, sigma);
}
"

include(Pkg.dir("Turing")*"/example-models/stan-models/school8-stan.data.jl")

global stanmodel, rc, sim1
# stanmodel = Stanmodel(name="schools8", model=eightschools);
stanmodel = Stanmodel(Sample(algorithm=Stan.Hmc(Stan.Static(0.75*5),Stan.diag_e(),0.75,0.0),
  save_warmup=true,adapt=Stan.Adapt(engaged=false)),
  num_samples=2000, num_warmup=0, thin=1,
  name="schools8", model=eightschools, nchains=1);

rc, sim1 = stan(stanmodel, schools8data, CmdStanDir=CMDSTAN_HOME, summary=false)

stan_time = get_stan_time("schools8")

println("Stan time: $stan_time")

describe(sim1)
