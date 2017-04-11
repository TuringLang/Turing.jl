
const simplegaussmodel = "
data {
  int<lower=0>  N;
  real x[N];
}
parameters {
  real<lower=0> s;
  real m;
}
model {
  s ~ inv_gamma(2,3);
  m ~ normal(0,sqrt(s));
    x ~ normal(m,sqrt(s));
}
"
