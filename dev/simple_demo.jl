using Turing

xs = Float64[0, 1, 0, 1, 0, 0, 0, 0, 0, 1]

@model beta begin
  @assume p1 ~ Beta(1, 1)
  for x in xs
    @observe x ~ Bernoulli(p1)
  end
  @predict p1
end

@time chain = sample(beta, HMC(10000, 0.01, 2))
mean(chain[:p1])


using Turing
using ForwardDiff: Dual
M = 9           # number of means
xs = [1.5, 2.0] # the observations
@model gauss_var begin
  ms = Vector{Dual}(M) # initialise an array to store means
  for i = 1:M
    @assume ms[i] ~ Normal(0, sqrt(2))  # define the mean
  end
  for i = 1:length(xs)
  	@observe xs[i] ~ Normal(mean(ms), sqrt(2)) # observe data points
  end
  @predict ms                  # ask predictions of s and m
end


t1 = time()
for _ = 1:100
  sample(gauss_var, HMC(100, 0.45, 5))
end
t = time() - t1
t / 100

# M = 3: 0.413e-1
# M = 9: 1.01
