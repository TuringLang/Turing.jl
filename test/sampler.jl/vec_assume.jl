include("../utility.jl")
using Distributions, Turing, Base.Test

N = 10
setchunksize(N)
# alg = HMCDA(2000, 0.65, 1.5)
alg = HMC(1000, 0.2, 4)

@model vdemo() = begin
  x = Vector{Real}(N)
  for i = 1:N
    x[i] ~ Normal(0, sqrt(4))
  end
end

t_loop = @elapsed res = sample(vdemo(), alg)


# Test for vectorize UnivariateDistribution
@model vdemo() = begin
  x = Vector{Real}(N)
  x ~ [Normal(0, 2)]
end

t_vec = @elapsed res = sample(vdemo(), alg)


@model vdemo() = begin
  x ~ MvNormal(zeros(N), 2 * ones(N))
end

t_mv = @elapsed res = sample(vdemo(), alg)

println("Time for")
println("  Loop : $t_loop")
println("  Vec  : $t_vec")
println("  Mv   : $t_mv")


# Transformed test
@model vdemo() = begin
  x = Vector{Real}(N)
  x ~ [InverseGamma(2, 3)]
end

sample(vdemo(), alg)
