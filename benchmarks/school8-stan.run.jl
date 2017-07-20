using Stan

# Model taken from https://github.com/goedman/Stan.jl/blob/master/Examples/Mamba/EightSchools/schools8.jl

include(Pkg.dir("Turing")*"/example-models/stan-models/school8-stan.model.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/school8-stan.data.jl")

global stanmodel, rc, sim
# stanmodel = Stanmodel(name="schools8", model=eightschools);
stanmodel = Stanmodel(Sample(algorithm=Stan.Hmc(Stan.Static(0.75*5),Stan.diag_e(),0.75,0.0),
  save_warmup=true,adapt=Stan.Adapt(engaged=false)),
  num_samples=2000, num_warmup=0, thin=1,
  name="schools8", model=eightschools, nchains=1);

rc, sim = stan(stanmodel, schools8data, CmdStanDir=CMDSTAN_HOME, summary=false)

stan_d = Dict()

for i = 1:8
  stan_d["eta[$i]"] = sim[:, ["eta.$i"], :].value[:]
  stan_d["theta[$i]"] = sim[:, ["theta.$i"], :].value[:]
end

stan_d["mu"] = sim[:, ["mu"], :].value[:]
stan_d["tau"] =sim[:, ["tau"], :].value[:]

for k = keys(stan_d)
  stan_d[k] = mean(stan_d[k])
end

# println("Stan time: $stan_time")

# describe(sim)
