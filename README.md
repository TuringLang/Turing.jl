# Turing.jl

[![Build Status](https://travis-ci.org/TuringLang/Turing.jl.svg?branch=master)](https://travis-ci.org/TuringLang/Turing.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/gp1xtxsc3971pwi6/branch/master?svg=true)](https://ci.appveyor.com/project/TuringLang/turing-jl/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/yebai/Turing.jl/badge.svg?branch=master)](https://coveralls.io/github/yebai/Turing.jl?branch=master)
[![Turing](http://pkg.julialang.org/badges/Turing_0.7.svg)](http://pkg.julialang.org/detail/Turing)
[![Turing](http://pkg.julialang.org/badges/Turing_0.6.svg)](http://pkg.julialang.org/detail/Turing)
[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](http://turing.ml/docs/)

News: **Turing.jl is now Julia 1.0 compatible now! Be aware that some things still might fail.**

**Turing.jl** is a Julia library for (_universal_) [probabilistic programming](https://en.wikipedia.org/wiki/Probabilistic_programming_language). Current features include:

- Universal probabilistic programming with an intuitive modelling interface
- Hamiltonian Monte Carlo (HMC) sampling for differentiable posterior distributions
- Particle MCMC sampling for complex posterior distributions involving discrete variables and stochastic control flows
- Gibbs sampling that combines particle MCMC,  HMC and many other MCMC algorithms

## Getting Started

To use Turing, you need to first install Julia. You will need to Julia 1.0 or greater, which you can get from [the official Julia website](http://julialang.org/downloads/). Once you have installing Julia, you need to add Turing to your Julia environment. Turing is an officially registered Julia package, so the following will install a stable version of Turing while inside Julia's package manager (press `]` from the REPL):

```julia
add Turing
```

If you want to use the latest version of Turing with some experimental features, you can try the following instead:

```julia
add Turing#master
test Turing
```

## Example

Here's a simple example showing the package in action:
```julia
using Turing
using Plots

# Define a simple Normal model with unknown mean and variance.
@model gdemo(x, y) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
  return s, m
end

#  Run sampler, collect results
c1 = sample(gdemo(1.5, 2), SMC(1000))
c2 = sample(gdemo(1.5, 2), PG(10,1000))
c3 = sample(gdemo(1.5, 2), HMC(1000, 0.1, 5))
c4 = sample(gdemo(1.5, 2), Gibbs(1000, PG(10, 2, :m), HMC(2, 0.1, 5, :s)))
c5 = sample(gdemo(1.5, 2), HMCDA(1000, 0.15, 0.65))
c6 = sample(gdemo(1.5, 2), NUTS(1000,  0.65))

# Summarise results (currently requires the master branch from MCMCChain)
describe(c3)

# Plot and save results
p = plot(c3)
savefig("gdemo-plot.png")
```
## Contributing

Turing was originally created and is now managed by Hong Ge. Current and past Turing team members include [Hong Ge](http://mlg.eng.cam.ac.uk/hong/), [Adam Scibior](http://mlg.eng.cam.ac.uk/?portfolio=adam-scibior), [Matej Balog](http://mlg.eng.cam.ac.uk/?portfolio=matej-balog), [Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/), [Kai Xu](http://mlg.eng.cam.ac.uk/?portfolio=kai-xu), [Emma Smith](https://github.com/evsmithx), [Emile Mathieu](http://emilemathieu.fr), [Martin Trapp](http://martint.blog).
You can see the full list of on Github: https://github.com/TuringLang/Turing.jl/graphs/contributors.

Turing is an open source project so if you feel you have some relevant skills and are interested in contributing then please do get in touch. See the [Contributing](http://turing.ml/docs/contributing/) page for details on the process. You can contribute by opening issues on Github or implementing things yourself and making a pull request. We would also appreciate example models written using Turing.

## Slack

Join [our channel](https://julialang.slack.com/messages/turing/) (`#turing`) in the Julia Slack chat for help, discussion, or general communication with the Turing team. If you do not already have an invitation to Julia's Slack, you can get one by going [here](https://slackinvite.julialang.org/).

## Citing Turing.jl ##
If you use **Turing** for your own research, please consider citing the following publication: Hong Ge, Kai Xu, and Zoubin Ghahramani: **Turing: Composable inference for probabilistic programming.** AISTATS 2018 [pdf](http://proceedings.mlr.press/v84/ge18b.html) [bibtex](https://dblp.org/rec/bib2/conf/aistats/GeXG18.bib)
