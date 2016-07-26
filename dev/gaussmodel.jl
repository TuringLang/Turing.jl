##########
# Turing #
##########
using Turing, Distributions, DualNumbers

xs = [1.5, 2.0]                            # the observations

@model gauss begin
  @assume s ~ InverseGamma(2, 3)           # define the variance
  @assume m ~ Normal(0, sqrt(s))           # define the mean
  for i = 1:length(xs)
  	@observe xs[i] ~ Normal(m, sqrt(s))    # observe data points
  end
  @predict s m                             # ask predictions of s and m
end

@time chain = sample(gauss, HMC(200, 0.15, 25))
# NOTE: s and m has N_Eff for different parameter settings. s need large ϵ and τ while m need small ones. This is worth to be mentioned in the dissertation.
using Mamba: Chains, summarystats
print(summarystats(Chains(chain[:s], names="s")))
print(summarystats(Chains(chain[:m], names="m")))

#     Mean       SD      Naive SE     MCSE       ESS
# s 2.1150453 2.0837529 0.04659413 0.11830779 310.2171
# m 1.1553526 0.86155516 0.019264959 0.022415524 1477.299



########
# Stan #
########
using Mamba, Stan

const gauss_data = [
  Dict("N" => 2, "xs" => [1.5, 2.0] )
]

const gauss_str = "
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
"

gauss = Stanmodel(name="gauss", model=gauss_str);
gauss_sim = stan(gauss, gauss_data, "$(pwd())/tmp/", CmdStanDir=CMDSTAN_HOME)

# Inference for Stan model: gauss_model
# 4 chains: each with iter=(1000,1000,1000,1000); warmup=(0,0,0,0); thin=(1,1,1,1); 4000 iterations saved.
#
# Warmup took (0.013, 0.022, 0.022, 0.024) seconds, 0.081 seconds total
# Sampling took (0.024, 0.040, 0.038, 0.038) seconds, 0.14 seconds total
#
#                 Mean     MCSE  StdDev        5%   50%   95%  N_Eff  N_Eff/s    R_hat
# lp__            -5.2  3.9e-02     1.2  -7.6e+00  -4.9  -4.1    857     6066  1.0e+00
# accept_stat__   0.91  3.1e-03    0.15   6.0e-01  0.96   1.0   2426    17178  1.0e+00
# stepsize__      0.68  3.2e-02   0.045   6.1e-01  0.70  0.73    2.0       14  2.8e+13
# treedepth__      2.1  1.2e-02    0.63   1.0e+00   2.0   3.0   2581    18278  1.0e+00
# n_leapfrog__     3.7  3.8e-02     2.0   1.0e+00   3.0   7.0   2834    20070  1.0e+00
# n_divergent__   0.00  0.0e+00    0.00   0.0e+00  0.00  0.00   4000    28324      nan
# s                2.1  5.8e-02     1.8   6.5e-01   1.6   5.3    958     6783  1.0e+00
# m                1.2  2.4e-02    0.82  -9.1e-02   1.2   2.5   1151     8150  1.0e+00
#
# Samples were drawn using hmc with nuts.
# For each parameter, N_Eff is a crude measure of effective sample size,
# and R_hat is the potential scale reduction factor on split chains (at
# convergence, R_hat=1).
