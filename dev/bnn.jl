# Toy example of multilayer perceptron:
# - learn the XOR function using two hidden layers

import ForwardDiff                    # for graident

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

# NN with only weights as variables
function nnw(args::Vector)
  w11, w12, w2 = args[1:3], args[4:6], args[7:9]
  x1 = [1; x[1]; x[2]]
  x2 = [1; sigmoid((w11' * x1)[1]); sigmoid((w12' * x1)[1])]
  y = [sigmoid((w2' * x2)[1])][1]
end

# Loss function: binary entropy
function loss(args::Vector)
  w11, w12, w2 = args[1:3], args[4:6], args[7:9]
  l = 0
  for i = 1:4
    x = xs[2*i-1:2*i]
    y = nn([x, w11, w12, w2])
    l += -(ys[i] * log2(y) + (1 - ys[i]) * log(1 - y))
  end
  l += 0.5 * (sum(w11 .* w11) + sum(w12 .* w12) + sum(w2 .* w2))
  return l
end

# xs = [[0; 0]; [0; 1]; [1; 0]; [1; 1]]
# ts = [0; 1; 1; 0]
# @model bnn begin
#   weights = TArray(Float64, 6)
#   @param σ ~ InverseGamma(2, 3)
#   for w in weights
#     @param w ~ Normal(0, sqrt(σ))
#   end
#   for i in 1:4
#     @observe ts[i] ~ Bernoulli(nn(xs[i], weights))
#   end
#   @predict weights
# end

# Find the gradient of loss function wrt to weights
∇loss = ForwardDiff.gradient(loss)

# Training data
xs = [Float64[0; 0]; Float64[0; 1]; Float64[1; 0]; Float64[1; 1]]
ys = [0; 1; 1; 0]

####################
# Gradient descend #
####################

# Learning rate
l_rate = 10

# Initialise weights
w11 = Float64[0.1; 0.1; 0.1]
w12 = Float64[0.1; 0.1; 0.1]
w2 = Float64[0.1; 0.1; 0.1]

# Print initial loss
println("Initial loss: ", loss([w11, w12, w2]))

# GD loop
for _ = 1:1000
  dw = ∇loss([w11, w12, w2])
  w2 -= l_rate * dw[7:9]
  dw = ∇loss([w11, w12, w2])  # why do we need to calculate the graident
  w11 -= l_rate * dw[1:3]     # each time we change part of the weight?
  dw = ∇loss([w11, w12, w2])
  w12 -= l_rate * dw[4:6]
end

# Print final loss
println("Final loss: ", loss([w11, w12, w2]))

# Print predictions
println("0; 0: ", nn([Float64[0; 0], w11, w12, w2]))
println("0; 1: ", nn([Float64[0; 1], w11, w12, w2]))
println("1; 0: ", nn([Float64[1; 0], w11, w12, w2]))
println("1; 1: ", nn([Float64[1; 1], w11, w12, w2]))

#########################
# Learning as inference #
#########################

include("hmc.jl")

G = w -> exp(-loss(w))
HMCSamples = HMCSampler(G, 500, 0.01, 10, 9)
s = HMCSamples
ws = [mean([s[i][1] for i = 1:500]);
      mean([s[i][2] for i = 1:500]);
      mean([s[i][3] for i = 1:500]);
      mean([s[i][4] for i = 1:500]);
      mean([s[i][5] for i = 1:500]);
      mean([s[i][6] for i = 1:500]);
      mean([s[i][7] for i = 1:500]);
      mean([s[i][8] for i = 1:500]);
      mean([s[i][9] for i = 1:500])]
w11, w12, w2 = ws[1:3], ws[4:6], ws[7:9]

# Print predictions
println("0; 0: ", nn([Float64[0; 0], w11, w12, w2]))
println("0; 1: ", nn([Float64[0; 1], w11, w12, w2]))
println("1; 0: ", nn([Float64[1; 0], w11, w12, w2]))
println("1; 1: ", nn([Float64[1; 1], w11, w12, w2]))
