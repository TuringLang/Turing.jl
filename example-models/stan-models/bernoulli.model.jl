# https://github.com/stan-dev/example-models/blob/master/basic_estimators/bernoulli.stan

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

@model bermodel(y) = begin
 theta ~ Beta(1,1)
 for n = 1:length(y)
   y[n] ~ Bernoulli(theta)
 end
 return theta
end
