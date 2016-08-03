##########
# Turing #
##########
using Turing, Distributions, DualNumbers, Gadfly, ForwardDiff
using Mamba: Chains, summarystats

function f(x, beta_0, beta_1, beta_2)
  return 1 / (1 + exp(-(beta_0 + beta_1 * x[1] + beta_2 * x[2])))
end

xs = Array[[1, 2], [2, 1], [-2, -1], [-1, -2]]
ts = [1, 1, 0, 0]

alpha = 0.25            # regularizatin term
var = sqrt(1 / alpha) # variance of the Gaussian prior
@model lr begin
  @assume beta_0 ~ Normal(0, var)
  @assume beta_1 ~ Normal(0, var)
  @assume beta_2 ~ Normal(0, var)
  for i = 1:4
    y = f(xs[i], beta_0, beta_1, beta_2)
    @observe ts[i] ~ Bernoulli(y)
  end
  @predict beta_0 beta_1 beta_2
end

@time chain1 = sample(lr, HMC(5000, 0.1, 5))
@time chain2 = sample(lr, HMC(5000, 0.1, 5))
@time chain3 = sample(lr, HMC(5000, 0.1, 5))
@time chain4 = sample(lr, HMC(5000, 0.1, 5))

mean([chain1[:beta_0] chain2[:beta_0] chain3[:beta_0] chain4[:beta_0]])
mean([chain1[:beta_1] chain2[:beta_1] chain3[:beta_1] chain4[:beta_1]])
mean([chain1[:beta_2] chain2[:beta_2] chain3[:beta_2] chain4[:beta_2]])

@time chain = sample(lr, PG(1000, 200))
mean(chain[:beta_0])
mean(chain[:beta_1])
mean(chain[:beta_2])

@time chain = sample(lr, SMC(5000))
mean(chain[:beta_0])
mean(chain[:beta_1])
mean(chain[:beta_2])



# Time in Turing

sample_nums = [10, 100, 500, 1000, 2000, 5000, 10000]

smc_time = []
for n = sample_nums
  t1 = time()
  for _ = 1:25
    sample(lr, SMC(n))
  end
  t2 = time()
  t = (t2 - t1) / 25
  push!(smc_time, t)
end

pg_time_1 = []
for n = [10, 100, 500, 1000]
  t1 = time()
  for _ = 1:10
    sample(lr, PG(10, n))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(pg_time_1, t)
end

pg_time_2 = []
for n = [10, 100, 500, 1000]
  t1 = time()
  for _ = 1:10
    sample(lr, PG(20, n))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(pg_time_2, t)
end

hmc_time_1 = []
for n = sample_nums
  t1 = time()
  for _ = 1:10
    sample(lr, HMC(n, 0.55, 5))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(hmc_time_1, t)
end

hmc_time_2 = []
for n = sample_nums
  t1 = time()
  for _ = 1:10
    sample(lr, HMC(n, 0.55, 15))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(hmc_time_2, t)
end

hmc_time_3 = []
# for n = sample_nums
for n = [5000, 10000]
  t1 = time()
  for _ = 1:10
    sample(lr, HMC(n, 1, 5))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(hmc_time_3, t)
end

sample_nums = [10, 100, 500, 1000, 2000, 5000, 10000]

hmc_time_1 = [0.12774250507354737,0.14173009395599365,0.5516953945159913,1.0368835926055908,2.4045027017593386,5.897187280654907,10.829419493675232]


hmc_time_2 = [0.028660011291503907,0.2859050989151001,1.4190002918243407,2.8486393213272097,5.77319450378418,14.528806400299072,29.45647838115692]


hmc_time_3 = [0.011547994613647462,0.13400509357452392,0.6510027170181274,1.0813661098480225,2.6234392881393434, 5.46103, 10.5817 ]

pg_time_1 = [0.2462191104888916,0.044436192512512206,0.16649439334869384,0.38031659126281736,0,0,0]

pg_time_2 = [0.20912668704986573,0.06705899238586426,0.3262843132019043,0.7031997919082642, 0, 0, 0]

smc_time = [0.09754419326782227,0.0046575927734375,0.017091922760009766,0.03385972023010254,0.07497027397155762,0.2280258846282959,0.5399847221374512]



using Gadfly
smc_layer = layer(x=sample_nums, y=smc_time, Geom.line, Theme(default_color=colorant"brown"))
pg_layer_1 = layer(x=sample_nums, y=pg_time_1, Geom.line, Theme(default_color=colorant"deepskyblue"))
pg_layer_2 = layer(x=sample_nums, y=pg_time_2, Geom.line, Theme(default_color=colorant"royalblue"))
hmc_layer_1 = layer(x=sample_nums, y=hmc_time_1, Geom.line, Theme(default_color=colorant"seagreen"))
hmc_layer_2 = layer(x=sample_nums, y=hmc_time_2, Geom.line, Theme(default_color=colorant"springgreen"))
hmc_layer_3 = layer(x=sample_nums, y=hmc_time_3, Geom.line, Theme(default_color=colorant"violet"))

p = plot(smc_layer, pg_layer_1, pg_layer_2, hmc_layer_1, hmc_layer_2, hmc_layer_3, Guide.ylabel("Time used (s)"), Guide.xlabel("#samples (n)"), Guide.manual_color_key("Legend", ["SMC(n)", "PG(10, n)", "PG(20, n)", "HMC(n, 0.55, 5)", "HMC(n, 0.55, 15)", "HMC(n, 1, 5)"], ["brown", "deepskyblue", "royalblue", "seagreen", "springgreen", "violet"]))

draw(PNG("/Users/kai/Turing/docs/report/withinturinglr.png", 4inch, 4inch), p)


# Time
t1 = time()
for _ = 1:10
  sample(lr, HMC(1000, 0.55, 5))
end
t2 = time()
t = (t2 - t1) / 10  #

# ESS and MCSE
mcse_0 = 0
mcse_1 = 0
mcse_2 = 0
ess_0 = 0
ess_1 = 0
ess_2 = 0
for _ = 1:100
  chain = sample(lr, HMC(1000, 0.55, 5))
  ss_0 = summarystats(Chains(chain[:beta_0]))
  ss_1 = summarystats(Chains(chain[:beta_1]))
  ss_2 = summarystats(Chains(chain[:beta_2]))
  ess_0 += ss_0.value[1, 5, 1]
  ess_1 += ss_1.value[1, 5, 1]
  ess_2 += ss_2.value[1, 5, 1]
  mcse_0 += ss_0.value[1, 4, 1]
  mcse_1 += ss_1.value[1, 4, 1]
  mcse_2 += ss_2.value[1, 4, 1]
end
ess_0 / 100 # 903.211435285377
ess_1 / 100 # 871.700557243748
ess_2 / 100 # 895.7831109654921
mcse_0 / 100 # 0.04878159108498346
mcse_1 / 100 # 0.046561007509454966
mcse_2 / 100 # 0.04489050778841781

########
# Stan #
########
using Mamba, Stan

const lr_data = [
  Dict(
    "N" => 4,
    "xs_1" => [1, 2, -2, -1],
    "xs_2" => [2, 1, -1, -2],
    "ts" => [1, 1, 0, 0]
  )
]

const lr_str = "
data {
  int N;
  real xs_1[N];
  real xs_2[N];
  int<lower=0,upper=1> ts[N];
}
parameters {
  real beta_0;
  real beta_1;
  real beta_2;
}
transformed parameters {
  real<lower=0,upper=1> ys[N];
  for (i in 1:N)
    ys[i] <- 1 / (1 + exp(-(beta_0 + beta_1 * xs_1[i] + beta_2 * xs_2[i])));
}
model {
  beta_0 ~ normal(0, 2);
  beta_1 ~ normal(0, 2);
  beta_2 ~ normal(0, 2);
  for (i in 1:N)
    ts[i] ~ bernoulli(ys[i]);
}
"

lr = Stanmodel(name="lr", model=lr_str);
lr_sim = stan(lr, lr_data, "$(pwd())/tmp/", CmdStanDir=CMDSTAN_HOME)
lr_sim_sub = beta_sim[1:1000, ["lp__", "stepsize__", "n_leapfrog__", "theta", "accept_stat__"], :]
describe(beta_sim_sub)



# Time
lr = Stanmodel(name="lr", model=lr_str, nchains=1)
t1 = time()
for _ = 1:10
  stan(lr, lr_data, "$(pwd())/tmp/", CmdStanDir=CMDSTAN_HOME)
end
t2 = time()
t = (t2 - t1) / 10  # 0.21471879482269288
# compile time = 5.082759714126587 - 0.21471879482269288

# ESS and MCSE
lr = Stanmodel(name="lr", model=lr_str, nchains=1)
mcse_0 = 0
ess_0 = 0
mcse_1 = 0
ess_1 = 0
mcse_2 = 0
ess_2 = 0
for _ = 1:100
  lr_pim = stan(lr, lr_data, "$(pwd())/tmp/", CmdStanDir=CMDSTAN_HOME)
  ss_0 = summarystats(Chains(reshape(lr_pim[1:1000, ["beta_0"], :].value, 1000, 1)))
  ss_1 = summarystats(Chains(reshape(lr_pim[1:1000, ["beta_1"], :].value, 1000, 1)))
  ss_2 = summarystats(Chains(reshape(lr_pim[1:1000, ["beta_2"], :].value, 1000, 1)))
  ess_0 += ss_0.value[1, 5, 1]
  mcse_0 += ss_0.value[1, 4, 1]
  ess_1 += ss_1.value[1, 5, 1]
  mcse_1 += ss_1.value[1, 4, 1]
  ess_2 += ss_2.value[1, 5, 1]
  mcse_2 += ss_2.value[1, 4, 1]
end
ess_0 / 100 #
mcse_0 / 100 #
ess_1 / 100 #
mcse_1 / 100 #
ess_2 / 100 #
mcse_2 / 100 #
