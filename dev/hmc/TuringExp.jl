using Turing, Distributions, DualNumbers

# Demo 1 - Univariate Gaussian
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
s = mean([d[:s] for d in chain[:samples]])
m = mean([d[:m] for d in chain[:samples]])

chain2 = sample(unigauss, IS(100))

chain3 = sample(unigauss, SMC(1000))

chain4 = sample(unigauss, PG(20, 30))

# Demo 2 - Mixture of Gaussians
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
  weights = [0; 0; 0; 0; 0; 0; 0; 0; 0]
  @assume σ ~ InverseGamma(2, 3)
  for w in weights
    @assume w ~ Normal(0, sqrt(σ))
  end
  for i in 1:4
    y = nn([xs[i], weights[1:3], weights[4:6], weights[7:9]])
    @observe ts[i] ~ Bernoulli(y)
  end
  @predict weights
end

chain = sample(bnn, HMC(1000, 0.01, 5))
