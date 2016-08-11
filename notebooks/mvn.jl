using Turing
using Distributions

seed_data = 0 #change to get another dataset
srand(seed_data)
d = 2 #dimension of the distribution

# hyperparameters and data for the model
Psi       = reshape(rand(d^2), d, d)^2 + d*eye(d)
mu        = rand(d) + 2.5
truemodel = MvNormal(mu, Psi)
data      = rand(truemodel, 10)

# the model
@model mvn begin
  @assume x ~ MvNormal(zeros(d), eye(d))
  for i = 1:size(data, 2)
    @observe data[:, i] ~ MvNormal(x,Psi)
  end
  @predict x
end

mvn_exact = posterior((MvNormal(zeros(d), eye(d)), Psi), MvNormal, data)

function mvn_evaluate(results)
  weights = map(x -> x.weight, results.value)
  samples = map(x -> x.value[:x], results.value)

  summary = Dict{Symbol,Any}()
  summary[:exact] = mvn_exact
  summary[:samples] = samples
  return summary
end
