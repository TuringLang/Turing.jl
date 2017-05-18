# https://github.com/stan-dev/example-models/blob/master/misc/moving-avg/stochastic-volatility.stan

using Distributions
using Turing
using HDF5, JLD

TPATH = Pkg.dir("Turing")

dataset = 1

sv_data = load(TPATH*"/nips-2017/sv/sv-data-$dataset.jld")["data"]

# model {
#   phi ~ uniform(-1,1);
#   sigma ~ cauchy(0,5);
#   mu ~ cauchy(0,10);
#   h[1] ~ normal(mu, sigma / sqrt(1 - phi * phi));
#   for (t in 2:T)
#     h[t] ~ normal(mu + phi * (h[t - 1] -  mu), sigma);
#   for (t in 1:T)
#     y[t] ~ normal(0, exp(h[t] / 2));
# }

@model sv_model(y) = begin
  T = length(y)
  ϕ ~ Uniform(-1, 1)
  σ ~ Truncated(Cauchy(0,5), 0, +Inf)
  μ ~ Cauchy(0, 10)
  h = tzeros(Real, T)
  if σ / sqrt(1 - ϕ^2) <= 0
    println(σ, ϕ)
  end
  h[1] ~ Normal(μ, σ / sqrt(1 - ϕ^2))
  y[1] ~ Normal(0, exp(h[1] / 2))
  for t = 2:T
    h[t] ~ Normal(μ + ϕ * (h[t-1] - μ) , σ)
    y[t] ~ Normal(0, exp(h[t] / 2))
  end
end

model_f = sv_model(data=sv_data[1])

sample_n = 10000

for run = 1:10

# Plot

# using Gadfly
# spl_colors = [colorant"#16a085", colorant"#8e44ad", colorant"#7f8c8d", colorant"#c0392b"]

# y = sv_data[1]["y"]
# plot(x=1:length(y),y=y,Geom.line)

setchunksize(550)
chain_nuts = sample(model_f, NUTS(sample_n, 0.65))
save(Pkg.dir("Turing")*"/nips-2017/sv/chain-nuts-data-$dataset-run-$run.jld", "chain", chain_nuts)
# sum(chain[:elapsed])
# lps = chain[:lp]
# l1 = layer(x=25:sample_n, y=-lps[25:end], Geom.line, Geom.line,Theme(default_color=spl_colors[1]))
# plot(l1)

setchunksize(5)
chain_gibbs = sample(model_f, Gibbs(sample_n, PG(50,1,:h),NUTS(1000,0.65,:ϕ,:σ,:μ)))
save(Pkg.dir("Turing")*"/nips-2017/sv/chain-gibbs-data-$dataset-run-$run.jld", "chain", chain_gibbs)
# sum(chain_gibbs[:elapsed])
# lps_gibbs = chain_gibbs[:lp]
# l2 = layer(x=2:sample_n, y=-lps_gibbs[2:end], Geom.line, Geom.line,Theme(default_color=spl_colors[2]))
# plot(l2)

# plot(l1, l2)

end
