using Turing
using Distributions
using Base.Test

@model bernoulli(y) begin
 theta ~ Beta(1,1)
 for n = 1:length(y)
   y[n] ~ Bernoulli(theta)
 end
 return theta
end


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
