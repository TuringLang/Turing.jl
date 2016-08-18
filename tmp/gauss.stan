data {
  int<lower=0> N;
  real xs[N];
}
parameters {
  real<lower=0> s;
  real m;
}
model {
  s ~ inv_gamma(2, 3);
  m ~ normal(0, sqrt(s));
    xs ~ normal(m, sqrt(s));
}