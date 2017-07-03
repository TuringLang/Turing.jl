using Distributions, Turing, Plots; gr()

# NOTE: chiange-point model is taken from http://www.mit.edu/~ilkery/papers/GibbsSampling.pdf
@model change_point(x) = begin
  N = length(x)
  n ~ Uniform(1, N)
  lam1 ~ Gamma(2, 1)
  lam2 ~ Gamma(2, 1)
  for i = 1:floor(Int, n)
    x[i] ~ Poisson(lam1)
  end
  for i = floor(Int, n)+1:N
    x[i] ~ Poisson(lam2)
  end
end

_N = 50
_n = floor(Int, rand(Uniform(1, _N)))
_lam1 = rand(Gamma(2, 1))
_lam2 = rand(Gamma(2, 1))
x = Vector{Int}(_N)
for i = 1:_n
  x[i] = rand(Poisson(_lam1))
end
for i = _n+1:_N
  x[i] = rand(Poisson(_lam2))
end
println("N = $_N, n = $_n, lam1 = $_lam1, lam2 = $_lam2")

plot(x, linetype=[:scatter], lab="data point")
plot!([_n], linetype=:vline, lab="change point")

modelf = change_point(x)

chn = sample(modelf, SMC(2000))

lam1_ = mean(chn[:lam1]); lam2_ = mean(chn[:lam2]); n_ = mean(chn[:n])

plot!([i <= n_ ? lam1_ : lam2_ for i = 1:_N], linetype=:steppre, lab="inference result")
title!("Change-point model (raw data & inference results)")
xlabel!("n"); ylabel!("counts")
