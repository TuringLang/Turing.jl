# Ref: https://github.com/stan-dev/example-models/blob/master/misc/moving-avg/stochastic-volatility.data.R

using Distributions

phi = 0.95;
sigma = 0.25;
beta = 0.6;
mu = 2 * log(beta);

T = 100;

h = Vector{Float64}(T);
h[1] = rand(Normal(mu, sigma / sqrt(1 - phi * phi)));
for t in 2:T
  h[t] = rand(Normal(mu + phi * (h[t-1] - mu), sigma));
end
y = Vector{Float64}(T);
for t in 1:T
  y[t] = rand(Normal(0, exp(h[t] / 2)));
end



const sv_data = [
  Dict(
  "T" => T,
  "y" => y
  )
]

const sv_exact_result = Dict(
  "ϕ" => phi,
  "σ" => sigma,
  "μ" => mu,
  "h" => h
)

using HDF5, JLD

save(Pkg.dir("Turing")*"/nips-2017/sv/sv_data.jld", "data", sv_data)
save(Pkg.dir("Turing")*"/nips-2017/sv/sv_exact_result.jld", "exact_result", sv_exact_result)
