data {
  int<lower=1> N;
  int<lower=0> y[N];
}
parameters {
  real<lower=0> alpha;
  real<lower=0> beta;
}
model {
  alpha ~ gamma(2,3);
  beta ~ beta(2,3);
  for (i in 1:N)
    y[i] ~ neg_binomial(alpha, beta);
}