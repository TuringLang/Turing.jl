# Test the multivariate distribution support for HMC sampler

using Turing, Distributions

# Define helper functions
function sigmoid(t)
  return 1 / (1 + e^(-t))
end

# Define NN flow
function nn(x, b1, w11, w12, w13, bo, wo)
  h = tanh([w11' * x + b1[1]; w12' * x + b1[2]; w13' * x + b1[3]])
  return sigmoid((wo' * h)[1] + bo)
end

# Generating training data
N = 20
M = round(Int64, N / 4)
x1s = rand(M) * 5
x2s = rand(M) * 5
xt1s = Array([[x1s[i]; x2s[i]] for i = 1:M])
append!(xt1s, Array([[x1s[i] - 6; x2s[i] - 6] for i = 1:M]))
xt0s = Array([[x1s[i]; x2s[i] - 6] for i = 1:M])
append!(xt0s, Array([[x1s[i] - 6; x2s[i]] for i = 1:M]))

xs = [xt1s; xt0s]
ts = [ones(M); ones(M); zeros(M); zeros(M)]

# Define model

alpha = 0.16            # regularizatin term
var = sqrt(1.0 / alpha) # variance of the Gaussian prior

@model bnn(ts) = begin
  b1 ~ MvNormal([0 ;0; 0], [var 0 0; 0 var 0; 0 0 var])
  w11 ~ MvNormal([0; 0], [var 0; 0 var])
  w12 ~ MvNormal([0; 0], [var 0; 0 var])
  w13 ~ MvNormal([0; 0], [var 0; 0 var])
  bo ~ Normal(0, var)

  wo ~ MvNormal([0; 0; 0], [var 0 0; 0 var 0; 0 0 var])
  for i = rand(1:N, 10)
    y = nn(xs[i], b1, w11, w12, w13, bo, wo)
    ts[i] ~ Bernoulli(y)
  end
  b1, w11, w12, w13, bo, wo
end

# Sampling
chain = sample(bnn(ts), HMC(10, 0.1, 5))
