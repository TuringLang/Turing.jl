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

@model negbinmodel(y) begin
  # α ~ Cauchy(0,10)
  # β ~ Cauchy(0,10)
  α ~ Gamma(2, 3)
  β ~ Beta(2, 3)
  for i = 1:length(y)
    y[i] ~ NegativeBinomial(α, β)  # α > 0, 0 < β < 1
  end
  return(α, β)
end

# NOTE: I change this model because Turing doesn't support "fore constrained varibales"
