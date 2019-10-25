# Turing.jl

[![Build Status](https://travis-ci.org/TuringLang/Turing.jl.svg?branch=master)](https://travis-ci.org/TuringLang/Turing.jl)
[![Build Status](https://dev.azure.com/yebai/TuringLang/_apis/build/status/TuringLang.Turing.jl?branchName=master)](https://dev.azure.com/yebai/TuringLang/_build/latest?definitionId=1&branchName=master)
[![Coverage Status](https://coveralls.io/repos/github/yebai/Turing.jl/badge.svg?branch=master)](https://coveralls.io/github/yebai/Turing.jl?branch=master)
[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://turing.ml/dev/docs/using-turing/)

**Turing.jl** is a Julia library for (_universal_) [probabilistic programming](https://en.wikipedia.org/wiki/Probabilistic_programming_language). Turing allows the user to write models using standard Julia syntax, and provide a wide range of sampling-based inference methods for solving problems across probabilistic machine learning, Bayesian statistics and data science etc. Compared to other probabilistic programming languages, Turing has a special focus on modularity, and decouples the modelling language (i.e. the compiler) and inference methods. This modular design, together with the use of a high-level numerical language Julia, makes Turing particularly extensible: new model families and inference methods can be easily added.

Current features include:

- General-purpose probabilistic programming with an intuitive modelling interface
- Robust, efficient [Hamiltonian Monte Carlo (HMC)](https://github.com/TuringLang/AdvancedHMC.jl) sampling for differentiable posterior distributions
- Particle MCMC sampling for complex posterior distributions involving discrete variables and stochastic control flows
- Compositional inference via Gibbs sampling that combines particle MCMC, HMC and random-walk MH (RWMH)

## Getting Started

Turing's home page, with links to everything you'll need to use Turing is:

https://turing.ml/dev/docs/using-turing/get-started


## What's changed recently?

See [releases](https://github.com/TuringLang/Turing.jl/releases).

## Want to contribute?

Turing was originally created and is now managed by Hong Ge. Current and past Turing team members include [Hong Ge](http://mlg.eng.cam.ac.uk/hong/), [Kai Xu](http://mlg.eng.cam.ac.uk/?portfolio=kai-xu), [Martin Trapp](http://martint.blog), [Mohamed Tarek](https://github.com/mohamed82008), [Cameron Pfiffer](https://business.uoregon.edu/faculty/cameron-pfiffer), [Tor Fjelde](http://retiredparkingguard.com/about.html).
You can see the full list of on Github: https://github.com/TuringLang/Turing.jl/graphs/contributors.

Turing is an open source project so if you feel you have some relevant skills and are interested in contributing then please do get in touch. See the [Contributing](http://turing.ml/docs/contributing/) page for details on the process. You can contribute by opening issues on Github or implementing things yourself and making a pull request. We would also appreciate example models written using Turing.

### Slack

Join [our channel](https://julialang.slack.com/messages/turing/) (`#turing`) in the Julia Slack chat for help, discussion, or general communication with the Turing team. If you do not already have an invitation to Julia's Slack, you can get one by going [here](https://slackinvite.julialang.org/).

## Related projects
- The Stan language for probabilistic programming - [Stan.jl](https://github.com/StanJulia/Stan.jl)
- Bare-bones implementation of robust dynamic Hamiltonian Monte Carlo methods - [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl)
- Comparing performance and results of mcmc options using Julia - [MCMCBenchmarks.jl](https://github.com/StatisticalRethinkingJulia/MCMCBenchmarks.jl)

## Citing Turing.jl ##
If you use **Turing** for your own research, please consider citing the following publication: Hong Ge, Kai Xu, and Zoubin Ghahramani: **Turing: a language for flexible probabilistic inference.** AISTATS 2018 [pdf](http://proceedings.mlr.press/v84/ge18b.html) [bibtex](https://github.com/TuringLang/Turing.jl/blob/master/CITATION.bib)
