# using Turing
# using Stan
#
# include(Pkg.dir("Turing")*"/benchmarks/benchmarkhelper.jl")
# include(Pkg.dir("Turing")*"/example-models/stan-models/MoC-stan.data.jl")
#
# @model nbmodel(K, V, M, N, z, w, doc, alpha, beta) = begin
#   theta ~ Dirichlet(alpha)
#   phi = Array{Any}(undef, K)
#   for k = 1:K
#     phi[k] ~ Dirichlet(beta)
#   end
#   for m = 1:M
#     z[m] ~ Categorical(theta)
#   end
#   for n = 1:N
#     w[n] ~ Categorical(phi[z[doc[n]]])
#   end
#   phi
# end
#
#
# bench_res = tbenchmark("NUTS(1000, 0.65)", "nbmodel", "data=nbstandata[1]")
# bench_res[4].names = ["phi[1]", "phi[2]", "phi[3]", "phi[4]"]
# logd = build_logd("Naive Bayes", bench_res...)
#
# include(Pkg.dir("Turing")*"/benchmarks/"*"MoC-stan.run.jl")
# logd["stan"] = stan_d
# logd["time_stan"] = nb_time
#
# print_log(logd)


using Turing
using Random
Random.seed!(129)
# Test for vectorize UnivariateDistribution
@model vdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0, sqrt(s))
  x ~ [Normal(m, sqrt(s))]
  # for i = 1:length(x)
  #   x[i] ~ Normal(m, sqrt.(s))
  # end
  return s, m
end

alg = HMC(250, 0.01, 5)
x = randn(1000)
res = sample(vdemo(x), alg)

# check_numerical(res, [:s, :m], [1, sum(x) / (1 + length(x))])

# Test for vectorize MultivariateDistribution

D = 2
@model vdemo2(x) = begin
  μ ~ MvNormal(zeros(D), ones(D))
  x ~ [MvNormal(μ, ones(D))]
end

alg = HMC(250, 0.01, 5)
res = sample(vdemo2(randn(D,1000)), alg)

# TODO: Test for vectorize MatrixDistribution
