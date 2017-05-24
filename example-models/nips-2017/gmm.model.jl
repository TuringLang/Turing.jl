using Distributions, Turing, HDF5, JLD, StatsFuns

TPATH = Pkg.dir("Turing")

@model gmm_gen(p, μ, σ) = begin
  z ~ Categorical(p)
  x ~ Normal(μ[z], σ[z])
end

make_norm_pdf(p, μ, σ) =
  x -> (pdf(Normal(μ[1], σ[1]), x) * p[1] + pdf(Normal(μ[2], σ[2]), x) * p[2] +
        pdf(Normal(μ[3], σ[3]), x) * p[3] + pdf(Normal(μ[4], σ[4]), x) * p[4] +
        pdf(Normal(μ[5], σ[5]), x) * p[5])

vn = Turing.VarName(gensym(), :x, "", 0)
@model gmm_gen_marg(p, μ, σ) = begin
  if isempty(vi)
    Turing.push!(vi, vn, 0, Normal(0,1), 0)
    x = rand(Uniform(-20,20))
  else
    x = vi[vn]
  end
  Turing.acclogp!(vi, log(make_norm_pdf(p, μ, σ)(x)))
end

M = 5
p = [ 0.2,  0.2,   0.2, 0.2,  0.2]
# μ = [   0,    1,     2, 3.5, 4.25] + 0.5 * collect(0:4)
μ = [   0,    1,     2, 3.5, 4.25] + 2.5 * collect(0:4)
s = [-0.5, -1.5, -0.75,  -2, -0.5]
σ = exp(s)
