using Distributions
using Turing
using Stan

include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
include(Pkg.dir("Turing")*"/example-models/stan-models/MoC-stan.data.jl")

@model nbmodel(K, V, M, N, z, w, doc, alpha, beta) = begin
  theta ~ Dirichlet(alpha)
  phi = Array{Any}(K)
  for k = 1:K
    phi[k] ~ Dirichlet(beta)
  end
  for m = 1:M
    z[m] ~ Categorical(theta)
  end
  for n = 1:N
    w[n] ~ Categorical(phi[z[doc[n]]])
  end
  phi
end


# bench_res = tbenchmark("NUTS(1000, 0.65)", "nbmodel", "data=nbstandata[1]")
# bench_res[4].names = ["phi[1]", "phi[2]", "phi[3]", "phi[4]"]
# logd = build_logd("Naive Bayes", bench_res...)
#
# include(Pkg.dir("Turing")*"/benchmarks/"*"MoC-stan.run.jl")
# logd["stan"] = stan_d
# logd["time_stan"] = nb_time
#
# print_log(logd)

samples = sample(nbmodel(data=nbstandata[1]), NUTS(1000, 0.65))
