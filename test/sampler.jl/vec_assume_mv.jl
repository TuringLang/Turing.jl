include("../utility.jl")
using Distributions, Turing, Base.Test

N = 5
beta = [0.5, 0.5]
setchunksize(N*length(beta))
alg = HMC(1000, 0.2, 4)

# Test for vectorize UnivariateDistribution
@model vdemo() = begin
  phi = Vector{Vector{Real}}(N)
  phi ~ [Dirichlet(beta)]
end

t_vec = @elapsed res_vec = sample(vdemo(), alg)

@model vdemo() = begin
  phi = Matrix(2,N)
  phi ~ [Dirichlet(beta)]
end

t_vec_mat = @elapsed res_vec_mat = sample(vdemo(), alg)

@model vdemo() = begin
  phi = Vector{Vector{Real}}(N)
  for i = 1:N
    phi[i] ~ Dirichlet(beta)
  end
end

t_loop = @elapsed res = sample(vdemo(), alg)

println("Time for")
println("  Loop : $t_loop")
println("  Vec  : $t_vec")
println("  Vec2 : $t_vec_mat")
