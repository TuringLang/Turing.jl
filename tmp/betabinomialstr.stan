data {                                  # data definition
  int<lower=0> N;
  int<lower=0,upper=1> obs[N];
}
parameters {                            # parameter definition
  real<lower=0,upper=1> theta;
}
model {                                 # model definition
  theta ~ beta(1,1);                    # define the prior
    obs ~ bernoulli(theta);             # observe data points
}