using Distributions, Turing, Mamba

N = 10
y = [0, 1, 0, 1, 0, 0, 0, 0, 0, 1]

@model bernoulli begin
  p ~ Beta(1,1)
  for i =1:N
    y[i] ~ Bernoulli(p)
  end
  return p
end

bdata = Dict(:N=>N, :y=>y)
c = sample(bernoulli, bdata, HMC(1000, 0.2, 5));
sim2 = Turing.TuringChains(c);

describe(sim1)
