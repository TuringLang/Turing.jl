using Mamba, Stan

old = pwd()
ProjDir = Pkg.dir("Stan", "Examples", "Bernoulli")
cd(ProjDir)

const bernoullistanmodel = "
data {
  int<lower=0> N;
  int<lower=0,upper=1> y[N];
}
parameters {
  real<lower=0,upper=1> theta;
}
model {
  theta ~ beta(1,1);
    y ~ bernoulli(theta);
}
"

stanmodel = Stanmodel(name="bernoulli", model=bernoullistanmodel);
stanmodel |> display

const bernoullidata = [
  Dict("N" => 10, "y" => [0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
]
println("Input observed data, an array of dictionaries:")
bernoullidata |> display
println()


sim1 = stan(stanmodel, bernoullidata, ProjDir, CmdStanDir=CMDSTAN_HOME)

describe(sim1)
