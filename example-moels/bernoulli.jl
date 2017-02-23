using Turing
using Distributions
using Base.Test

@model bernoulli() begin
 theta ~ Beta(1,1)
 for n = 1:length(y)
   y[n] ~ Bernoulli(theta)
 end
 return theta
end


data = Dict(:y => [0,1,0,0,0,0,0,0,0,1])
sample(bernoulli, data, HMC(1000, 0.1, 5))

"
data {
  int<lower=0> N;
  int<lower=0,upper=1> y[N];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(1,1);
  for (n in 1:N)
    y[n] ~ bernoulli(theta);
}
"
