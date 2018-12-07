# Test the importance sampler
# The test is based on running a reference implementation of a model with the same seed as the importance sampler.
# Both predictions and logevidence are compared for equality.

using Turing
using Test
using Random
using StatsFuns

function reference(n :: Int)
  logweights = zeros(Float64, n)
  samples = Array{Dict{Symbol,Any}}(undef, n)
  for i = 1:n
    samples[i] = reference()
    logweights[i] = samples[i][:logweight]
  end
  logevidence = logsumexp(logweights) - log(n)
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

let n = 10
  @model normal() = begin
    a ~ Normal(4,5)
    3 ~ Normal(a,2)
    b ~ Normal(a,1)
    1.5 ~ Normal(b,2)
    a, b
  end

  alg = IS(n)
  seed = 0

  _f = normal();
  for i=1:100
    Random.seed!(seed)
    exact = reference(n)
    Random.seed!(seed)
    tested = sample(_f, alg)
    for i = 1:n
      @test exact[:samples][i][:a] == tested[:samples][i][:a]
      @test exact[:samples][i][:b] == tested[:samples][i][:b]
      @test exact[:logweights][i]  == tested[:logweights][i]
    end
    @test exact[:logevidence] .== first(tested[:logevidence])
  end
end
