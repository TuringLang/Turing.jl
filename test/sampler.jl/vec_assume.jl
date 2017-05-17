include("../utility.jl")
using Distributions, Turing, Base.Test

N = 100
setchunksize(N)
# alg = HMCDA(2000, 0.65, 1.5)
alg = HMC(2000, 0.2, 4)

@model vdemo() = begin
  x = Vector{Real}(N)
  for i = 1:N
    x[i] ~ Normal(0, sqrt(4*i))
  end
end

t_loop = @elapsed res = sample(vdemo(), alg)
@test_approx_eq_eps mean(mean(res[:x])) 0 0.1


# Test for vectorize UnivariateDistribution
@model vdemo() = begin
  x = Vector{Real}(N)
  x ~ [Normal(0, 2)]
end

t_vec = @elapsed res = sample(vdemo(), alg)
@test_approx_eq_eps mean(mean(res[:x])) 0 0.1


@model vdemo() = begin
  x ~ MvNormal(zeros(N), 2 * ones(N))
end

t_mv = @elapsed res = sample(vdemo(), alg)
@test_approx_eq_eps mean(mean(res[:x])) 0 0.1

println("Time for")
println("  Loop : $t_loop")
println("  Vec  : $t_vec")
println("  Mv   : $t_mv")
