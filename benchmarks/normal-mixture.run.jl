using Distributions
using Turing
using Stan

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/normal-mixture-stan.data.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/normal-mixture.model.jl")

# NOTE: I only run a sub-set of the data as running the whole is quite slow
tbenchmark("Gibbs(10, HMC(1, 0.05, 1, :theta), PG(50, 1, :k), HMC(1, 0.2, 3, :mu))", "nmmodel", "simplenormalmixturestandata[1][\"y\"][1:100]")

bench_res = tbenchmark("Gibbs(1000, HMC(1, 0.05, 1, :theta), PG(50, 1, :k), HMC(1, 0.2, 3, :mu))", "nmmodel", "simplenormalmixturestandata[1][\"y\"][1:100]")
logd = build_logd("Simple Gaussian Mixture Model", bench_res...)

include("normal-mixture-stan.run.jl")
logd["stan"] = Dict("theta" => mean(nm_theta), "mu[1]" => mean(nm_mu_1), "mu[2]" =>mean(nm_mu_2))
logd["time_stan"] = nm_time

print_log(logd, ["theta", "mu[1]", "mu[2]"])
