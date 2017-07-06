# https://github.com/goedman/Stan.jl/blob/master/Examples/Mamba/Binormal/binormal.jl

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")

using Stan
using Mamba: describe

const binorm = "
transformed data {
  matrix[2,2] Sigma;
  vector[2] mu;
  mu[1] <- 0.0;
  mu[2] <- 0.0;
  Sigma[1,1] <- 1.0;
  Sigma[2,2] <- 1.0;
  Sigma[1,2] <- 0.10;
  Sigma[2,1] <- 0.10;
}
parameters {
  vector[2] y;
}
model {
    y ~ multi_normal(mu,Sigma);
}
"

global stanmodel, rc, sim1, sim
# stanmodel = Stanmodel(name="binormal", model=binorm, Sample(save_warmup=true));
stanmodel = Stanmodel(Sample(algorithm=Stan.Hmc(Stan.Static(0.5*5),Stan.diag_e(),0.5,0.0),
  save_warmup=true,adapt=Stan.Adapt(engaged=false)),
  num_samples=2000, num_warmup=0, thin=1,
  name="binormal", model=binorm, nchains=1);

rc, sim1 = stan(stanmodel, CmdStanDir=CMDSTAN_HOME, summary=false)

if rc == 0
  ## Subset Sampler Output
  sim = sim1[1:size(sim1, 1), ["lp__", "y.1", "y.2"], 1:size(sim1, 3)]

  describe(sim)

  stan_time = get_stan_time("binormal")

end # cd

using Turing

@model binormal() = begin
  y ~ MvNormal(zeros(2), [1.0 0.1; 0.1 1.0])
end

chn = sample(binormal(), HMC(2000,0.5,5))

describe(chn)

turing_time = sum(chn[:elapsed])

println("Stan time  : $stan_time")
println("Turing time: $turing_time")
