using Distributions
using Turing
using Turing: invlogit

# Load the first 2000 points of dataset a9a (from adult UCI)
include("lr_helper.jl")
x, y = readlrdata()
n, d = size(x)

# Shuffle and create train and tests datasets
n_train = Int(round(0.8*n))
n_test = Int(n - n_train)
indexes = randperm(n)[1:n]
index_train = indexes[1:n_train]
index_test = indexes[n_train+1:end]
x_train = x[index_train, :]
y_train = y[index_train]
x_test = x[index_test, :]
y_test = y[index_test]

logistic(x::Real) = invlogit(x)

# Bayesian logistic regression (LR) with Laplace prior as 5.2 in SGLD paper
batch_size = 10
@model lr_nuts(x, y, d, n) = begin
  α ~ Laplace()
  β = Array{ForwardDiff.Dual}(d)
  for k in 1:d
    β[k] ~ Laplace()
  end
  for i = 1:10
    j = rand(1:n)
    y′ = logistic(α + dot(x[j,:], β))
    y[j] ~ Bernoulli(y′)
  end

end

chain = sample(lr_nuts(x_train, y_train, d, n_train), SGLD(400, 0.5))
# chain = sample(lr_nuts(x_train, y_train, d, n_train), SGHMC(400, 0.01, 0.5))

# Build arameters' posterior estimator by wheighting by step size (cf 4.2 in paper)
β_estimator = zeros(Float32, d)
for i in 1:d
  index = find(chain.names .== string("β[",i,"]"))
  β_estimator[i] = sum(chain.value[:,index] .* chain[:lf_eps]) / sum(chain[:lf_eps])
end

α_estimator = sum(chain[:α] .* chain[:lf_eps]) / sum(chain[:lf_eps])

# Compute accuracy
right = 0
for j in 1:n_test
  y′ = logistic(α_estimator + dot(x_test[j,:], β_estimator))
  y′ = Int32(round(y′))
  right += (y′ == y_test[j])
end

print("Accuracy: ", round(right/n_test,2))
