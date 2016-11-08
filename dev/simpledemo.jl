using Turing, Distributions, Base.Test

xs = [1, 2]

@model test begin
  @assume μ ~ Gamma(1, 2)
  for i = 1:length(xs)
    @observe xs[i] ~ Normal(μ, 1)
  end
  @predict μ
end

h = HMC(2000, 0.2, 10)

res = sample(test, h)
mean(res[:μ])



res2 = sample(test, PG(50, 100))
mean(res2[:μ])
