include("../utility.jl")
using Distributions, Turing


setchunksize(100)
# Test for vectorize UnivariateDistribution
@model vdemo() = begin
  x = Vector{Real}(100)
  x ~ [Normal(0, sqrt(4))]
  # for i = 1:100
  #   x[i] ~ Normal(0, sqrt(4))
  # end
end

alg = HMC(1000, 0.2, 4)
res = sample(vdemo(), alg)
println(mean(mean(res[:x])))
