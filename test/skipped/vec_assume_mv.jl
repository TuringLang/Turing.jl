
using Turing, Test

N = 10
beta = [0.5, 0.5]
setchunksize(N*length(beta))
alg = HMC(0.2, 4)

# Test for vectorize UnivariateDistribution
@model vdemo() = begin
  phi = Vector{Vector{Real}}(undef, N)
  @> phi ~ Dirichlet(beta)
end

ch_vec, t_vec, m_vec, gctime, memallocs = @timed res_vec = sample(vdemo(), alg)

@model vdemo() = begin
  phi = Matrix(undef, 2, N)
  @. phi ~ Dirichlet(beta)
end

ch_vec_mat, t_vec_mat, m_vec_mat, gctime, memallocs = @timed res_vec_mat = sample(vdemo(), alg)

@model vdemo() = begin
  phi = Vector{Vector{Real}}(undef, N)
  for i = 1:N
    phi[i] ~ Dirichlet(beta)
  end
end

ch_loop, t_loop, m_loop, gctime, memallocs = @timed res = sample(vdemo(), alg)

println("Time for")
println("  Loop : $(sum(ch_loop[:elapsed]))")
println("  Vec  : $(sum(ch_vec[:elapsed]))")
println("  Vec2 : $(sum(ch_vec_mat[:elapsed]))")

println("Mem for")
println("  Loop : $m_loop")
println("  Vec  : $m_vec")
println("  Vec2 : $m_vec_mat")
