using Mamba, Stan

const betabinomial_data = [              # the observations
  Dict("N" => 10, "obs" => [0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
]

const betabinomial_str = "
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
"

betabinomial = Stanmodel(name="betabinomial", model=betabinomial_str)
betabinomial_sim = stan(betabinomial, betabinomial_data, "$(pwd())/tmp/", CmdStanDir=CMDSTAN_HOME)
