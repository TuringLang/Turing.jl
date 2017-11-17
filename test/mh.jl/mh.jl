include("../utility.jl")

using Distributions
using Turing
using Base.Test
srand(125)
x = [1.5 2.0]

@model mhtest(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  for i in 1:length(x)
    x[i] ~ Normal(m, sqrt(s))
  end
  s, m
end

model = mhtest(x)

# MH with prior as proposal
res = sample(model, MH(100))
check_numerical(res,
[:s, :m], [49/24, 7/6]
)

# MH with Gaussian proposal
GaussianKernel(var) = (x) -> Normal(x, sqrt(var))
check_numerical(
  sample(model, MH(500, (:s, GaussianKernel(1.0)), (:m, GaussianKernel(1.0)))),
  [:s, :m], [49/24, 7/6]
)

# MH within Gibbs
check_numerical(
  sample(model, Gibbs(100, MH(5, :m), MH(5, :s))),
  [:s, :m], [49/24, 7/6]
)
