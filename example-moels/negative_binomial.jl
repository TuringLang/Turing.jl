using Turing
using Distributions
using Base.Test

@model negative_binomial() begin
 alpha ~ Cauchy(0,10)
 beta ~ Cauchy(0,10)
 for i = 1:length(y)
   y[i] ~ NegativeBinomial(alpha, beta)
 end
 return alpha, beta
end


# Produce an error.
data = Dict(:y => [0, 1, 4, 0, 2, 2, 5, 0, 1])
sample(negative_binomial, data, HMC(1000, 0.1, 5))


"data {
  int<lower=1> N;
  int<lower=0> y[N];
}
parameters {
  real<lower=0> alpha;
  real<lower=0> beta;
}
model {
  alpha ~ cauchy(0,10);
  beta ~ cauchy(0,10);
  for (i in 1:N)
    y[i] ~ neg_binomial(alpha, beta);
}
"
