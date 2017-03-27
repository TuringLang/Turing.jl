# Test the importance sampler
# The test is based on running a reference implementation of a model with the same seed as the importance sampler.
# Both predictions and logevidence are compared for equality.

using Turing
using Distributions
using Base.Test

function logsum(xs :: Vector{Float64})
  largest = maximum(xs)
  ys = map(x -> exp(x - largest), xs)
  result = log(sum(ys)) + largest
  return result
end

function reference(n :: Int)
  logweights = zeros(Float64, n)
  samples = Array{Dict{Symbol,Any}}(n)
  for i = 1:n
    samples[i] = reference()
    logweights[i] = samples[i][:logweight]
  end
  logevidence = logsum(logweights) - log(n)
  results = Dict{Symbol,Any}()
  results[:logevidence] = logevidence
  results[:logweights] = logweights
  results[:samples] = samples
  return results
end

function reference()
  x = rand(Normal(4,5))
  y = rand(Normal(x,1))
  log_lik = logpdf(Normal(x,2), 3) + logpdf(Normal(y,2), 1.5)
  d = Dict()
  d[:logweight] = log_lik
  d[:a] = x
  d[:b] = y
  return d
end

@model normal begin
  a ~ Normal(4,5)
  3 ~ Normal(a,2)
  b ~ Normal(a,1)
  1.5 ~ Normal(b,2)
  a, b
end

n = 10
alg = IS(n)
seed = 0

for i=1:100
  srand(seed)
  exact = reference(n)
  srand(seed)
  tested = @sample(normal(), alg)
  for i = 1:n
    @test exact[:samples][i][:a] == tested[:samples][i][:a]
    @test exact[:samples][i][:b] == tested[:samples][i][:b]
    @test exact[:logweights][i]  == tested[:logweights][i]
  end
  @test exact[:logevidence] == tested[:logevidence]
end
