using Distributions
using Turing
using Stan

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/bernoulli.data.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/bernoulli..model.jl")

bench_res = tbenchmark("HMC(1000, 0.25, 5)", "bermodel", "berdata")
logd = build_logd("Bernoulli Model", bench_res...)


include(Pkg.dir("Turing")*"/benchmarks/"*"bernoulli-stan.run.jl")
logd["stan"] = Dict("theta" => mean(theta_stan))
logd["time_stan"] = ber_time

print_log(logd)
