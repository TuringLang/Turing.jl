include("hmc.jl")

### model start

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

xs = [[0; 0]; [0; 1]; [1; 0]; [1; 1]]
ts = [0; 1; 1; 0]

function likelihood(argList::Vector)
  μ = argList[1]
  σ = argList[2]

  pdf = hmcNormal(μ, σ)

  xs = [1, 1, 1, 1.25, 0.75]
  lik = 1

  for x in xs
    lik *= hmcpdf(pdf, x)
  end

  return lik
end

function prior(argList::Vector)
  μ = argList[1]
  σ = argList[2]
  pri = 1 * hmcpdf(hmcInverseGamma(2, 3), σ) * hmcpdf(hmcNormal(0, sqrt(σ)), μ)
  return pri
end

function posterior(argList::Vector)
  post = likelihood(argList) * prior(argList)
  return post
end
