# include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/binormal-stan.model.jl")

using Stan
# using Mamba: describe

global stanmodel, rc, sim1, sim, stan_time
# stanmodel = Stanmodel(name="binormal", model=binorm, Sample(save_warmup=true));
stanmodel = Stanmodel(Sample(algorithm=Stan.Hmc(Stan.Static(0.5*5),Stan.diag_e(),0.5,0.0),
  save_warmup=true,adapt=Stan.Adapt(engaged=false)),
  num_samples=2000, num_warmup=0, thin=1,
  name="binormal", model=binorm, nchains=1);

rc, sim1 = stan(stanmodel, CmdStanDir=CMDSTAN_HOME, summary=false)

if rc == 0
  ## Subset Sampler Output
  sim = sim1[1:size(sim1, 1), ["lp__", "y.1", "y.2"], 1:size(sim1, 3)]

  # describe(sim)

  

end # cd
