##########
# Turing #
##########
using Turing, Distributions, DualNumbers
using Mamba: Chains, summarystats

xs = Float64[0, 1, 0, 1, 0, 0, 0, 0, 0, 1]

@model beta begin
  @assume p ~ Beta(1, 1)
  for x in xs
    @observe x ~ Bernoulli(p)
  end
  @predict p
end

@time chain1 = sample(beta, HMC(1000, 0.1, 2))
@time chain2 = sample(beta, HMC(1000, 0.1, 2))
@time chain3 = sample(beta, HMC(1000, 0.1, 2))
@time chain4 = sample(beta, HMC(1000, 0.1, 2))

print(summarystats(Chains([chain1[:p] chain2[:p] chain3[:p] chain4[:p]], names=["p", "p", "p", "p"])))

@time chain = sample(beta, PG(500, 200))

mean(chain[:p])

@time chain = sample(beta, SMC(4000))

mean(chain[:p])


# Time in Turing

sample_nums = [10, 100, 500, 1000, 2000, 5000, 10000]

smc_time = []
for n = sample_nums
  t1 = time()
  for _ = 1:25
    sample(beta, SMC(n))
  end
  t2 = time()
  t = (t2 - t1) / 25
  push!(smc_time, t)
end

pg_time_1 = []
for n = sample_nums
  t1 = time()
  for _ = 1:10
    sample(beta, PG(10, n))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(pg_time_1, t)
end

pg_time_2 = []
for n = sample_nums
  t1 = time()
  for _ = 1:10
    sample(beta, PG(20, n))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(pg_time_2, t)
end

hmc_time_1 = []
for n = sample_nums
  t1 = time()
  for _ = 1:10
    sample(beta, HMC(n, 0.01, 2))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(hmc_time_1, t)
end

hmc_time_2 = []
for n = sample_nums
  t1 = time()
  for _ = 1:10
    sample(beta, HMC(n, 0.01, 10))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(hmc_time_2, t)
end

hmc_time_3 = []
for n = sample_nums
  t1 = time()
  for _ = 1:10
    sample(beta, HMC(n, 0.03, 2))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(hmc_time_3, t)
end

sample_nums = [10, 100, 500, 1000, 2000, 5000, 10000]

hmc_time_1 = [0.0015016794204711914,0.015407109260559082,0.08041720390319824,0.15822238922119142,0.3266536951065063,0.860090708732605,1.8383234024047852]

hmc_time_2 = [0.003940796852111817,0.05217530727386475,0.24969069957733153,0.5556180000305175,1.0059500932693481,2.4296218156814575,4.835002803802491]

hmc_time_3 = [0.002090311050415039,0.01472010612487793,0.0862234115600586,0.16342289447784425,0.3292788028717041,0.968375301361084,1.868330192565918]

pg_time_1 = [0.04356000423431396,0.05189070701599121,0.2793883800506592,0.5344091892242432,0.972931694984436,3.325191783905029,8.516942191123963]

pg_time_2 = [0.008345198631286622,0.08994572162628174,0.4455733060836792,0.8887288093566894,1.7949350118637084,5.356746983528137,12.388906693458557]

smc_time = [0.050951480865478516,0.005121879577636719,0.025275478363037108,0.07786175727844238,0.1483304786682129,0.43594951629638673,0.8674785614013671]


using Gadfly
smc_layer = layer(x=sample_nums, y=smc_time, Geom.line, Theme(default_color=colorant"brown"))
pg_layer_1 = layer(x=sample_nums, y=pg_time_1, Geom.line, Theme(default_color=colorant"deepskyblue"))
pg_layer_2 = layer(x=sample_nums, y=pg_time_2, Geom.line, Theme(default_color=colorant"royalblue"))
hmc_layer_1 = layer(x=sample_nums, y=hmc_time_1, Geom.line, Theme(default_color=colorant"seagreen"))
hmc_layer_2 = layer(x=sample_nums, y=hmc_time_2, Geom.line, Theme(default_color=colorant"springgreen"))
hmc_layer_3 = layer(x=sample_nums, y=hmc_time_3, Geom.line, Theme(default_color=colorant"violet"))

p = plot(smc_layer, pg_layer_1, pg_layer_2, hmc_layer_1, hmc_layer_2, hmc_layer_3, Guide.ylabel("Time used (s)"), Guide.xlabel("#samples (n)"), Guide.manual_color_key("Legend", ["SMC(n)", "PG(10, n)", "PG(20, n)", "HMC(n, 0.01, 2)", "HMC(n, 0.01, 10)", "HMC(n, 0.03, 2)"], ["brown", "deepskyblue", "royalblue", "seagreen", "springgreen", "violet"]))

draw(PNG("/Users/kai/Turing/docs/report/withinturingbeta.png", 4inch, 4inch), p)


# Time
t1 = time()
for _ = 1:25
  sample(beta, HMC(1000, 0.05, 3))
end
t2 = time()
t = (t2 - t1) / 25  # 0.23458239555358887


# ESS and MCSE
mcse_p = 0
ess_p = 0
for _ = 1:25
  chain = sample(beta, HMC(1000, 0.1, 2))
  ss_p = summarystats(Chains(chain[:p]))
  ess_p += ss_p.value[1, 5, 1]
  mcse_p += ss_p.value[1, 4, 1]
end
ess_p / 100 # 208.39164810063977
mcse_p / 100 # 0.0010638107443104346


########
# Stan #
########
using Mamba, Stan

const beta_data = [
  Dict("N" => 10, "y" => [0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
]

const beta_str = "
data {
  int<lower=0> N;
  int<lower=0,upper=1> y[N];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(1, 1);
    y ~ bernoulli(theta);
}
"

beta = Stanmodel(name="beta", model=beta_str);
beta_sim = stan(beta, beta_data, "$(pwd())/tmp/", CmdStanDir=CMDSTAN_HOME)
beta_sim_sub = beta_sim[1:1000, ["lp__", "stepsize__", "n_leapfrog__", "theta", "accept_stat__"], :]
describe(beta_sim_sub)




# Time
beta = Stanmodel(name="beta", model=beta_str, nchains=1)
t1 = time()
for _ = 1:10
  stan(beta, beta_data, "$(pwd())/tmp/", CmdStanDir=CMDSTAN_HOME)
end
t2 = time()
t = (t2 - t1) / 10  # 0.17233469486236572
# compile time = 5.833784985542297 - 0.17233469486236572 = 5.661450290679931

# ESS and MCSE
beta = Stanmodel(name="beta", model=beta_str, nchains=1)
mcse_p = 0
ess_p = 0
for _ = 1:100
  beta_pim = stan(beta, beta_data, "$(pwd())/tmp/", CmdStanDir=CMDSTAN_HOME)
  ss_p = summarystats(Chains(reshape(beta_pim[1:1000, ["theta"], :].value, 1000, 1)))
  ess_p += ss_p.value[1, 5, 1]
  mcse_p += ss_p.value[1, 4, 1]
end
ess_p / 100 # 459.6555754405945
mcse_p / 100 # 0.006534593385605037
