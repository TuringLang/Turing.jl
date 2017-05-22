using Distributions
using Turing
using HDF5, JLD

TPATH = Pkg.dir("Turing")

@model gmm_gen(p, μ, σ) = begin
  z ~ Categorical(p)
  x ~ Normal(μ[z], σ[z])
end

make_norm_pdf(μ, σ) =
  x -> (pdf(Normal(μ[1], σ[1]), x) + pdf(Normal(μ[2], σ[2]), x) +
        pdf(Normal(μ[3], σ[3]), x) + pdf(Normal(μ[4], σ[4]), x) +
        pdf(Normal(μ[5], σ[5]), x)) / 5

vn = Turing.VarName(gensym(), :x, "", 0)
@model gmm_gen_marg(p, μ, σ) = begin
  if isempty(vi)
    Turing.push!(vi, vn, 0, Normal(0,1), 0)
    x = rand(Uniform(-20,20))
  else
    x = vi[vn]
  end
  Turing.acclogp!(vi, log(make_norm_pdf(μ, σ)(x)))
end

dev = 2.5
M = 5
p = [ 0.2,  0.2,   0.2, 0.2,  0.2]
μ = [   0,    1,     2, 3.5, 4.25] + dev*collect(0:4)
s = [-0.5, -1.5, -0.75,  -2, -0.5]
σ = exp(s)

N = 100000
K = 500

for i = 1:1
  chain_gibbs = sample(gmm_gen(p, μ, σ), Gibbs(round(Int,N/K), PG(5, 1, :z), HMC(K-1, 0.2, 8, :x); thin=false))
  save(TPATH*"/nips-2017/gmm/gibbs-chain-$dev-$i.jld", "chain", chain_gibbs)

  chain_nuts = sample(gmm_gen_marg(p, μ, σ), NUTS(N, 0.65))
  save(TPATH*"/nips-2017/gmm/nuts-chain-$dev-$i.jld", "chain", chain_nuts)
end
