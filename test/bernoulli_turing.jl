using Distributions, Turing

N = 10
y = [0, 1, 0, 1, 0, 0, 0, 0, 0, 1]

@model bernoulli(y) begin
  @assume p ~ Beta(1,1)
  for i =1:N
    @observe y[i] ~ Bernoulli(p)
  end
  @predict p
end

c = @sample(bernoulli(y), HMC(1000, 0.2, 5))

mean(c[:p])
var(c[:p])
