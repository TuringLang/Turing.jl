include("bernoulli.data.jl")
include("bernoulli.model.jl")

alg_str = "HMC(1000, 0.25, 5)"
alg = eval(parse(alg_str))
ber_sim = sample(bermodel(berdata), alg)

logd = Dict(
  "name" => "Bernoulli Model",
  "engine" => "$alg_str",
  "time" => time,
  "mem" => mem,
  "turing" => Dict("theta" => mean(ber_sim[:theta])),
  # "analytic" => Dict("theta" => ???),
  "stan" => Dict("theta" => mean(theta_stan))
)

print_log(logd)
