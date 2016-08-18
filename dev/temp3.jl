using Turing, Distributions, DualNumbers, Gadfly

# Generate synthesised data
xs = rand(Normal(0.5, 1), 20)

# Define model
@model unigauss begin
  priors = zeros(2)
  @assume priors[1] ~ InverseGamma(2, 3)
  @assume priors[2] ~ Normal(0, sqrt(priors[1]))
  for x in xs
    @observe x ~ Normal(priors[2], sqrt(priors[1]))
  end
  @predict priors
end

chain = sample(unigauss, PG(20, 20))


type TestT
  abb   ::    Any
  function TestT(abb)
    new(abb)
  end
end

t = TestT(123)
print(t)
convert(::AbstractString, x::TestT) =
string(123)
