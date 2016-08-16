using Turing, Distributions, DualNumbers, Gadfly, ForwardDiff
# using Mamba: Chains, summarystats



function sigmoid(t)
  return 1 / (1 + e^(-t))
end

function nn(x, wi1, wi2, wh1, wh2, wo)
  # wi = weight_in, wh = weight_hidden, wo = weight_out
  h1 = tanh([wi1' * x; wi2' * x])
  h2 = tanh([wh1' * h1; wh2' * h1])
  wo = sigmoid((wo' * h2)[1])
end



xs = Array[[1; 1], [-1; -1], [1; -1], [-1; 1]]
ts = [1; 1; 0; 0]

alpha = 0.25            # regularizatin term
var = sqrt(1.0 / alpha) # variance of the Gaussian prior
@model bnn begin
  @assume wi1 ~ MvNormal([0; 0], [var 0; 0 var])
  @assume wi2 ~ MvNormal([0; 0], [var 0; 0 var])
  @assume wh1 ~ MvNormal([0; 0], [var 0; 0 var])
  @assume wh2 ~ MvNormal([0; 0], [var 0; 0 var])
  @assume wo ~ MvNormal([0; 0], [var 0; 0 var])
  for i = 1:4
    y = nn(xs[i], wi1, wi2, wh1, wh2, wo)
    @observe ts[i] ~ Bernoulli(y)
  end
  @predict wi1 wi2 wh1 wh2 wo
end

@time chain = sample(bnn, HMC(2500, 0.01, 5))



function predict(x, chain)
  n = length(chain[:wi1])
  wi1 = chain[:wi1]
  wi2 = chain[:wi2]
  wh1 = chain[:wh1]
  wh2 = chain[:wh2]
  wo = chain[:wo]
  return mean([nn(x, wi1[i], wi2[i], wh1[i], wh2[i], wo[i]) for i in 1:n])
end

[predict(xs[i], chain) for i = 1:4]
