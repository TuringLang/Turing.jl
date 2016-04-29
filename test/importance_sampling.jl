using Turing
using Distributions
using Base.Test

# testing the importance sampler
# the test is based on running a reference implementation of a model with the same seed as importance sampler
# both predictions and logevidence are compared for equality

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
  d[:x] = x
  d[:y] = y
  return d
end

@model normal begin
  @assume x ~ Normal(4,5)
  @observe 3 ~ Normal(x,2)
  @assume y ~ Normal(x,1)
  @observe 1.5 ~ Normal(y,2)
  @predict x y
end

n = 10
alg = IS(n)
seed = 0

for i=1:100
  srand(seed)
  exact = reference(n)
  srand(seed)
  tested = sample(normal, alg)
  for i = 1:n
    @test exact[:samples][i][:x] == tested[:samples][i][:x]
    @test exact[:samples][i][:y] == tested[:samples][i][:y]
    @test exact[:logweights][i]  == tested[:logweights][i]
  end
  @test exact[:logevidence] == tested[:logevidence]
end
