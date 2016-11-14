#=
We also need some helper functions in this demo.
sigmoid() is simply the sigmoid function.
nn() is the input-output mapping of the neural network, which will be shown later.
predict() makes Bayesian prediction by averaging the NN outputs of all sample weights in the chain provided. This predicit function may run slowly because of the large amount of samples.
=#

# Sigmod function
function sigmoid(t)
  return 1 / (1 + e^(-t))
end

# The Neural Net
function nn(x, b1, w11, w12, w13, w14, b2, w21, w22, bo, wo)
  # wi = weight_in, wh = weight_hidden, wo = weight_out
  h1 = tanh([w11' * x + b1[1]; w12' * x + b1[2]; w13' * x + b1[3]; w14' * x + b1[4]])
  h2 = tanh([w21' * h1 + b2[1]; w22' * h1 + b2[2]])
  return sigmoid((wo' * h2)[1] + bo)
end

# Bayesian prediction - averaging predcitions from all samples
function predict(x, chain, ratio=1.0)
    n = Int64(length(chain[:b1]) * ratio)
  b1 = chain[:b1][1:n]
  w11 = chain[:w11][1:n]
  w12 = chain[:w12][1:n]
  w13 = chain[:w13][1:n]
  w14 = chain[:w14][1:n]
  b2 = chain[:b2][1:n]
  w21 = chain[:w21][1:n]
  w22 = chain[:w22][1:n]
  bo = chain[:bo][1:n]
  wo = chain[:wo][1:n]
  return mean([nn(x, b1[i], w11[i], w12[i], w13[i], w14[i], b2[i], w21[i], w22[i], bo[i], wo[i]) for i in 1:10:n])
end