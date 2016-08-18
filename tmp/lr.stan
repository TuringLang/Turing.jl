data {
  int N;
  real xs_1[N];
  real xs_2[N];
  int<lower=0,upper=1> ts[N];
}
parameters {
  real beta_0;
  real beta_1;
  real beta_2;
}
transformed parameters {
  real<lower=0,upper=1> ys[N];
  for (i in 1:N)
    ys[i] <- 1 / (1 + exp(-(beta_0 + beta_1 * xs_1[i] + beta_2 * xs_2[i])));
}
model {
  beta_0 ~ normal(0, 2);
  beta_1 ~ normal(0, 2);
  beta_2 ~ normal(0, 2);
  for (i in 1:N)
    ts[i] ~ bernoulli(ys[i]);
}