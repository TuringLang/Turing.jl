##########
# Turing #
##########
using Turing, Distributions, DualNumbers
using Mamba: Chains, summarystats

xs = [1.5, 2.0]                            # the observations

@model gauss begin
  @assume s ~ InverseGamma(2, 3)           # define the variance
  @assume m ~ Normal(0, sqrt(s))           # define the mean
  for i = 1:length(xs)
  	@observe xs[i] ~ Normal(m, sqrt(s))    # observe data points
  end
  @predict s m                             # ask predictions of s and m
end

@time chain1 = sample(gauss, HMC(1000, 0.55, 4))
@time chain2 = sample(gauss, HMC(1000, 0.55, 4))
@time chain3 = sample(gauss, HMC(1000, 0.55, 4))
@time chain4 = sample(gauss, HMC(1000, 0.55, 4))

# NOTE: s and m has N_Eff for different parameter settings. s need large ϵ and τ while m need small ones. This is worth to be mentioned in the dissertation.

print(summarystats(Chains([chain1[:s] chain2[:s] chain3[:s] chain4[:s]], names=["s", "s", "s", "s"])))
print(summarystats(Chains([chain1[:m] chain2[:m] chain3[:m] chain4[:m]], names=["m", "m", "m", "m"])))

#     Mean       SD      Naive SE     MCSE       ESS
# s 1.9786154 1.6126516 0.02549826 0.064673827 621.7617
# m 1.1564340 0.81268557 0.012849687 0.008232733 4000

@time chain1 = sample(gauss, PG(500, 100))
@time chain2 = sample(gauss, PG(500, 100))
@time chain3 = sample(gauss, PG(500, 100))
@time chain4 = sample(gauss, PG(500, 100))

mean([chain1[:s]; chain2[:s]; chain3[:s]; chain4[:s]])
mean([chain1[:m]; chain2[:m]; chain3[:m]; chain4[:m]])

@time chain1 = sample(gauss, SMC(1000))
@time chain2 = sample(gauss, SMC(1000))
@time chain3 = sample(gauss, SMC(1000))
@time chain4 = sample(gauss, SMC(1000))

mean([chain1[:s]; chain2[:s]; chain3[:s]; chain4[:s]])
mean([chain1[:m]; chain2[:m]; chain3[:m]; chain4[:m]])

# Time in Turing
t1 = time()
for _ = 1:25
  sample(gauss, HMC(1000, 0.2, 5))
end
t2 = time()
t = (t2 - t1) / 25

sample_nums = [10, 100, 500, 1000, 2000, 5000, 10000]

smc_time = []
for n = sample_nums
  t1 = time()
  for _ = 1:25
    sample(gauss, SMC(n))
  end
  t2 = time()
  t = (t2 - t1) / 25
  push!(smc_time, t)
end

pg_time_1 = []
for n = sample_nums
  t1 = time()
  for _ = 1:10
    sample(gauss, PG(10, n))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(pg_time_1, t)
end

pg_time_2 = []
for n = sample_nums
  t1 = time()
  for _ = 1:10
    sample(gauss, PG(20, n))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(pg_time_2, t)
end
print(pg_time_2)

hmc_time_1 = []
for n = sample_nums
  t1 = time()
  for _ = 1:10
    sample(gauss, HMC(n, 0.05, 2))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(hmc_time_1, t)
end

hmc_time_2 = []
for n = sample_nums
  t1 = time()
  for _ = 1:10
    sample(gauss, HMC(n, 0.05, 10))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(hmc_time_2, t)
end

hmc_time_3 = []
for n = sample_nums
  t1 = time()
  for _ = 1:10
    sample(gauss, HMC(n, 0.5, 2))
  end
  t2 = time()
  t = (t2 - t1) / 10
  push!(hmc_time_3, t)
end

sample_nums = [10, 100, 500, 1000, 2000, 5000, 10000]

hmc_time_1 = [0.004650497436523437,0.04269888401031494,0.20954039096832275,0.4710076093673706,0.938780689239502,2.242368292808533,3.5772396087646485]

hmc_time_2 = [0.015126204490661621,0.15580008029937745,0.7784370899200439,1.5663403034210206,2.908810591697693,7.821655702590943,17.936883306503297]


hmc_time_3 = [0.009491491317749023,0.060298705101013185,0.29063079357147215,0.5673642873764038,1.1188395023345947,2.6560792922973633,5.046795201301575]

pg_time_1 = [0.2000460147857666,0.029438495635986328,0.13155040740966797,0.2751434087753296,0.5456831932067872,1.4251347064971924,3.046298289299011]
pg_time_2 = [0.007204389572143555,0.06006867885589599,0.31555991172790526,0.5549521923065186,1.0265091896057128,2.766718101501465,5.308166193962097]

smc_time = [0.0004083251953125,0.0035467529296875,0.015932798385620117,0.03243380546569824,0.06348095893859863,0.1791096019744873,0.45574819564819335]

using Gadfly
smc_layer = layer(x=sample_nums, y=smc_time, Geom.line, Theme(default_color=colorant"brown"))
pg_layer_1 = layer(x=sample_nums, y=pg_time_1, Geom.line, Theme(default_color=colorant"deepskyblue"))
pg_layer_2 = layer(x=sample_nums, y=pg_time_2, Geom.line, Theme(default_color=colorant"royalblue"))
hmc_layer_1 = layer(x=sample_nums, y=hmc_time_1, Geom.line, Theme(default_color=colorant"seagreen"))
hmc_layer_2 = layer(x=sample_nums, y=hmc_time_2, Geom.line, Theme(default_color=colorant"springgreen"))
hmc_layer_3 = layer(x=sample_nums, y=hmc_time_3, Geom.line, Theme(default_color=colorant"violet"))

p = plot(smc_layer, pg_layer_1, pg_layer_2, hmc_layer_1, hmc_layer_2, hmc_layer_3, Guide.ylabel("Time used (s)"), Guide.xlabel("#samples (n)"), Guide.manual_color_key("Legend", ["SMC(n)", "PG(10, n)", "PG(20, n)", "HMC(n, 0.05, 2)", "HMC(n, 0.05, 20)", "HMC(n, 0.5, 2)"], ["brown", "deepskyblue", "royalblue", "seagreen", "springgreen", "violet"]))

draw(PDF("/Users/kai/Turing/docs/report/withinturing.pdf", 4inch, 4inch), p)


# Time
t1 = time()
for _ = 1:10
  sample(gauss, HMC(1000, 0.55, 4))
end
t2 = time()
t = (t2 - t1) / 10  # 0.5942052125930786

# ESS and MCSE
mcse_s = 0
mcse_m = 0
ess_s = 0
ess_m = 0
for _ = 1:100
  chain = sample(gauss, HMC(1000, 0.55, 4))
  ss_s = summarystats(Chains(chain[:s]))
  ss_m = summarystats(Chains(chain[:m]))
  ess_s += ss_s.value[1, 5, 1]
  ess_m += ss_m.value[1, 5, 1]
  mcse_s += ss_s.value[1, 4, 1]
  mcse_m += ss_m.value[1, 4, 1]
end
ess_s / 100 # 181.16478750018905
ess_m / 100 # 825.040657070573
mcse_s / 100 # 0.1952434962425414
mcse_m / 100 # 0.02743143131913111


########
# Stan #
########
using Mamba, Stan

const gauss_data = [
  Dict("N" => 2, "xs" => [1.5, 2.0] )
]

const gauss_str = "
data {
  int<lower=0> N;
  real xs[N];
}
parameters {
  real<lower=0> s;
  real m;
}
model {
  s ~ inv_gamma(2, 3);
  m ~ normal(0, sqrt(s));
    xs ~ normal(m, sqrt(s));
}
"

gauss = Stanmodel(name="gauss", model=gauss_str)
gauss_sim = stan(gauss, gauss_data, "$(pwd())/tmp/", CmdStanDir=CMDSTAN_HOME)
gauss_sim_sub = gauss_sim[1:1000, ["lp__", "stepsize__", "n_leapfrog__", "s", "m", "accept_stat__"], :]
describe(gauss_sim_sub)

# Time
gauss = Stanmodel(name="gauss", model=gauss_str, nchains=1)
t1 = time()
for _ = 1:10
  stan(gauss, gauss_data, "$(pwd())/tmp/", CmdStanDir=CMDSTAN_HOME)
end
t2 = time()
t = (t2 - t1) / 10  # 0.355244517326355

# ESS and MCSE
gauss = Stanmodel(name="gauss", model=gauss_str, nchains=1)
mcse_s = 0
mcse_m = 0
ess_s = 0
ess_m = 0
for _ = 1:100
  gauss_sim = stan(gauss, gauss_data, "$(pwd())/tmp/", CmdStanDir=CMDSTAN_HOME)
  ss_s = summarystats(Chains(reshape(gauss_sim[1:1000, ["s"], :].value, 1000, 1)))
  ss_m = summarystats(Chains(reshape(gauss_sim[1:1000, ["m"], :].value, 1000, 1)))
  ess_s += ss_s.value[1, 5, 1]
  ess_m += ss_m.value[1, 5, 1]
  mcse_s += ss_s.value[1, 4, 1]
  mcse_m += ss_m.value[1, 4, 1]
end
ess_s / 100 # 356.01590116934824
ess_m / 100 # 378.54305932355135
mcse_s / 100 # 0.10649510480431454
mcse_m / 100 # 0.04642232215932677









# different influcne on s and m
ess_s = 0
ess_m = 0
for _ = 1:25
  chain = sample(gauss, HMC(1000, 0.1, 10))
  ss_s = summarystats(Chains(chain[:s]))
  ss_m = summarystats(Chains(chain[:m]))
  ess_s += ss_s.value[1, 5, 1]
  ess_m += ss_m.value[1, 5, 1]
end
ess_s / 25
ess_m / 25
