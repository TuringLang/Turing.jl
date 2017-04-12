include("simple-normal-mixture.data.jl")
include("simple-normal-mixture.model.jl")

# NOTE: I only run a sub-set of the data as running the whole is quite slow
bench_res = tbenchmark("Gibbs(500, PG(20, 1, :k), HMC(1, 1.5, 2, :mu, :theta))", "nmmodel", "y[1:100]")
logd = build_logd("Simple Gaussian Mixture Model", bench_res...)

logd["stan"] = Dict("theta" => mean(nm_theta), "mu[1]" => mean(nm_mu_1), "mu[2]" =>mean(nm_mu_2))

print_log(logd, ["theta", "mu[1]", "mu[2]"])
