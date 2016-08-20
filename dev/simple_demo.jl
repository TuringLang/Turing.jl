using Turing

xs = Float64[0, 1, 0, 1, 0, 0, 0, 0, 0, 1]

@model beta begin
  @assume p ~ Beta(1, 1)
  for x in xs
    @observe x ~ Bernoulli(p)
  end
  @predict p
end

@time chain = sample(beta, HMC(10, 0.1, 2))


p = dBeta(1, 1)
