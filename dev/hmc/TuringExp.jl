##############################
# Demo - Univariate Gaussian #
##############################
using Turing, Distributions, DualNumbers

xs = rand(Normal(0.5, 4), 50)

@model unigauss begin
  @assume s ~ InverseGamma(2, 3)
  @assume m ~ Normal(0, sqrt(s))
  for x in xs
    @observe x ~ Normal(m, sqrt(s))
  end
  @predict s m
end

chain = sample(unigauss, HMC(1000, 0.05, 5))
s = sqrt(mean([d[:s] for d in chain[:samples]]))
m = mean([d[:m] for d in chain[:samples]])

chain2 = sample(unigauss, IS(100))

chain3 = sample(unigauss, SMC(1000))
s3 = sqrt(mean([d.weight * d.value[:s] for d in chain3.value]))
m3 = mean([d.weight * d.value[:m] for d in chain3.value])

chain4 = sample(unigauss, PG(20, 30))


################################
# Demo - Multivariate Gaussian #
################################
using Turing, Distributions, DualNumbers, PDMats

xs = rand(MvNormal(Vector{Float64}([1, 1]),
                   PDMat(Array{Float64,2}([1 0; 0 1]))),
          50)

@model multigauss begin
  @assume m ~ MvNormal([1, 1], [4 0; 0 4])
  for x in xs
    @observe x ~ MvNormal(m, [1 0; 0 1])
  end
  @predict m
end

chain = sample(multigauss, HMC(250, 0.001, 5))
m = mean([d[:m] for d in chain[:samples]])

####################################################
# Demo - Bayesian Neural Nets with a Single Neuron #
####################################################
using Turing, Distributions, DualNumbers, Gadfly

# Helper function for the single neuron bnn
function singley(x, w0, w1, w2)
  return 1 / (1 + exp(-(w0 + w1 * x[1] + w2 * x[2])))
end

# Training data
xs = Array[[1, 2], [2, 1], [-2, -1], [-1, -2]]
ts = [1, 1, 0, 0]

# Define the model
α = 0.01          # regularizatin term
σ = sqrt(1 / α)   # variance of the Gaussian prior
@model singlebnn begin
  @assume w0 ~ Normal(0, σ)
  @assume w1 ~ Normal(0, σ)
  @assume w2 ~ Normal(0, σ)
  for i = 1:4
    y = singley(xs[i], w0, w1, w2)
    @observe ts[i] ~ Bernoulli(y)
  end
  @predict w0 w1 w2
end

# Sample the model
chain = sample(singlebnn, HMC(1000, 0.15, 1))
# chain2 = sample(singlebnn, SMC(25))
# w0 = mean([d.weight * d.value[:w0] for d in chain2.value])
# w1 = mean([d.weight * d.value[:w1] for d in chain2.value])
# w2 = mean([d.weight * d.value[:w2] for d in chain2.value])
# y = Float64[singley(xs[i], w0, w1, w2) for i = 1:4]

# Helper function for predicting
function singlepredict(x, chain)
  return mean([singley(x, d[:w0], d[:w1], d[:w2]) for d in chain[:samples]])
end

# Compute predctions
y = Float64[singlepredict(xs[i], chain) for i = 1:4]

# Plot predictions
singledata_layer_1 = layer(x=Float64[1, 2], y=Float64[2, 1], Geom.point, Theme(default_color=colorant"red"))
singledata_layer_2 = layer(x=Float64[-1, -2], y=Float64[-2, -1], Geom.point, Theme(default_color=colorant"blue"))
singlepredictions_layer = layer(z=(x,y) -> singlepredict([x, y], chain), x=linspace(-4,4,25), y=linspace(-4,4,25), Geom.contour)
singlepredictions_plot = plot(singledata_layer_1, singledata_layer_2, singlepredictions_layer, Guide.xlabel("dim 1"), Guide.ylabel("dim 2"),Guide.title("Predictions of the Single Neuron BNN"), Coord.cartesian(xmin=-4, xmax=4, ymin=-4, ymax=4))

# Output plot
draw(PNG("/Users/kai/Turing/docs/demo/singlebnn.png", 6inch, 5.5inch), singlepredictions_plot)

###################################################
# Demo - Bayesian Neural Nets with 1 Hidden Layer #
####################################################
using Turing, Distributions, DualNumbers, Gadfly

# Activation function
function sigmoid(a)
  1 / (1 + exp(-a))
end

# NN with 1 hidden layer
function hiddeny(x, w11, w12, w2)
  x1 = [1; x[1]; x[2]]
  x2 = [1; sigmoid((w11' * x1)[1]); sigmoid((w12' * x1)[1])]
  y = [sigmoid((w2' * x2)[1])][1]
end

# Training data
xs = Array[[0, 0], [0, 1], [1, 0], [1, 1]]
ts = [0, 1, 1, 0]

# Define the model
α = 0.01          # regularizatin term
σ = sqrt(1 / α)   # variance of the Gaussian prior
mu = [0, 0, 0]
Σ = [σ 0 0; 0 σ 0; 0 0 σ]
@model hiddenbnn begin
  @assume w11 ~ MvNormal(mu, Σ)
  @assume w12 ~ MvNormal(mu, Σ)
  @assume w2 ~ MvNormal(mu, Σ)
  for i in 1:4
    y = hiddeny(xs[i], w11, w12, w2)
    @observe ts[i] ~ Bernoulli(y)
  end
  @predict w11 w12 w2
end

# Sample the model
chain = sample(hiddenbnn, HMC(500, 0.1, 1))

# Helper function for predicting
function hiddenpredict(x, chain)
  return mean([hiddeny(x, d[:w11], d[:w12], d[:w2]) for d in chain[:samples]])
end

# Compute predctions
y = Float64[hiddenpredict(xs[i], chain) for i = 1:4]
