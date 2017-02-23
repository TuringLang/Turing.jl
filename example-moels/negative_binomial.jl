using Turing
using Distributions
using Base.Test

@model negative_binomial(y) begin
 alpha ~ Cauthy(0,10)
 beta ~ Cauthy(0,10)
 for i = 1:length(y)
   y[i] ~ NegativeBinomial(alpha, beta)
 end
 return alpha, beta
end

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
