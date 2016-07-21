# Demo - Univariate Gaussian
using Turing, Distributions, DualNumbers

xs = rand(Normal(0.5, 4), 500)

@model unigauss begin
  @assume s ~ InverseGamma(2, 3)
  @assume m ~ Normal(0, sqrt(s))
  for x in xs
    @observe x ~ Normal(m, sqrt(s))
  end
  @predict s m
end

chain = sample(unigauss, HMC(1000, 0.01, 5))   # HMC(n_samples, lf_size, lf_num)
s = sqrt(mean([d[:s] for d in chain[:samples]]))
m = mean([d[:m] for d in chain[:samples]])

chain2 = sample(unigauss, IS(100))

chain3 = sample(unigauss, SMC(1000))

chain4 = sample(unigauss, PG(20, 30))



# Demo - Multivariate Gaussian
using Turing, Distributions, DualNumbers, PDMats

xs = rand(MvNormal(Vector{Float64}([1, 1]),
                   PDMat(Array{Float64,2}([1 0; 0 1]))),
          100)

@model multigauss begin
  @assume m ~ MvNormal([1, 1], [4 0; 0 4])
  for x in xs
    @observe x ~ MvNormal(m, [1 0; 0 1])
  end
  @predict m
end

chain = sample(multigauss, HMC(250, 0.001, 5))
m = mean([d[:m] for d in chain[:samples]])


# Demo - Mixture of Gaussians
using Turing, Distributions, DualNumbers

# Activation function
function sigmoid(a)
  1 / (1 + exp(-a))
end

# NN with 1 hidden layer
function nn(args::Vector)
  x, w11, w12, w2 = args[1:2], args[3:5], args[6:8], args[9:11]
  x1 = [1; x[1]; x[2]]
  x2 = [1; sigmoid((w11' * x1)[1]); sigmoid((w12' * x1)[1])]
  y = [sigmoid((w2' * x2)[1])][1]
end

xs = Array[[0;0], [0;1], [1;0], [1;1]]
ts = [0; 1; 1; 0]
@model bnn begin
  @assume σ ~ InverseGamma(2, 3)
  @assume w1 ~ Normal(0, sqrt(σ))
  @assume w2 ~ Normal(0, sqrt(σ))
  @assume w3 ~ Normal(0, sqrt(σ))
  @assume w4 ~ Normal(0, sqrt(σ))
  @assume w5 ~ Normal(0, sqrt(σ))
  @assume w6 ~ Normal(0, sqrt(σ))
  @assume w7 ~ Normal(0, sqrt(σ))
  @assume w8 ~ Normal(0, sqrt(σ))
  @assume w9 ~ Normal(0, sqrt(σ))
  for i in 1:4
    y = nn([xs[i], w1, w2, w3, w4, w5, w6, w7, w8, w9])
    @observe ts[i] ~ Bernoulli(y)
  end
  @predict w1 w2 w3 w4 w5 w6 w7 w8 w9
end

chain = sample(bnn, HMC(1000, 0.01, 5))
w1 = mean([d[:w1] for d in chain[:samples]])
w2 = mean([d[:w1] for d in chain[:samples]])
w3 = mean([d[:w1] for d in chain[:samples]])
w4 = mean([d[:w1] for d in chain[:samples]])
w5 = mean([d[:w1] for d in chain[:samples]])
w6 = mean([d[:w1] for d in chain[:samples]])
w7 = mean([d[:w1] for d in chain[:samples]])
w8 = mean([d[:w1] for d in chain[:samples]])
w9 = mean([d[:w1] for d in chain[:samples]])
