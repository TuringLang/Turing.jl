# Turing.jl

[![Build Status](https://travis-ci.org/TuringLang/Turing.jl.svg?branch=master)](https://travis-ci.org/TuringLang/Turing.jl)
[![Build Status](https://dev.azure.com/yebai/TuringLang/_apis/build/status/TuringLang.Turing.jl?branchName=master)](https://dev.azure.com/yebai/TuringLang/_build/latest?definitionId=1&branchName=master)
[![Coverage Status](https://coveralls.io/repos/github/yebai/Turing.jl/badge.svg?branch=master)](https://coveralls.io/github/yebai/Turing.jl?branch=master)
[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://turing.ml/dev/docs/using-turing/)

**Turing.jl** is a Julia library for (_universal_) [probabilistic programming](https://en.wikipedia.org/wiki/Probabilistic_programming_language). Turing allows the user to write models in standard Julia syntax, and provide a wide range of sampling-based inference methods for solving problems across probabilistic machine learning, Bayesian statistics and data science etc. Since Turing is implemented in pure Julia code, its compiler and inference methods are amendable for hacking: new model families and inference methods can be easily added.

Current features include:

- Universal probabilistic programming with an intuitive modelling interface
- Hamiltonian Monte Carlo (HMC) sampling for differentiable posterior distributions
- Particle MCMC sampling for complex posterior distributions involving discrete variables and stochastic control flows
- Gibbs sampling that combines particle MCMC,  HMC and many other MCMC algorithms

## Getting Started

Turing's home page, with links to everything you'll need to use Turing is:

https://turing.ml/dev/docs/using-turing/get-started


## What's changed recently?

See [releases](https://github.com/TuringLang/Turing.jl/releases).

## Want to contribute?

Turing was originally created and is now managed by Hong Ge. Current and past Turing team members include [Hong Ge](http://mlg.eng.cam.ac.uk/hong/), [Adam Scibior](http://mlg.eng.cam.ac.uk/?portfolio=adam-scibior), [Matej Balog](http://mlg.eng.cam.ac.uk/?portfolio=matej-balog), [Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/), [Kai Xu](http://mlg.eng.cam.ac.uk/?portfolio=kai-xu), [Emma Smith](https://github.com/evsmithx), [Emile Mathieu](http://emilemathieu.fr), [Martin Trapp](http://martint.blog).
You can see the full list of on Github: https://github.com/TuringLang/Turing.jl/graphs/contributors.

Turing is an open source project so if you feel you have some relevant skills and are interested in contributing then please do get in touch. See the [Contributing](http://turing.ml/docs/contributing/) page for details on the process. You can contribute by opening issues on Github or implementing things yourself and making a pull request. We would also appreciate example models written using Turing.

### Slack

Join [our channel](https://julialang.slack.com/messages/turing/) (`#turing`) in the Julia Slack chat for help, discussion, or general communication with the Turing team. If you do not already have an invitation to Julia's Slack, you can get one by going [here](https://slackinvite.julialang.org/).

## Citing Turing.jl ##
If you use **Turing** for your own research, please consider citing the following publication: Hong Ge, Kai Xu, and Zoubin Ghahramani: **Turing: a language for flexible probabilistic inference.** AISTATS 2018 [pdf](http://proceedings.mlr.press/v84/ge18b.html) [bibtex](https://github.com/TuringLang/Turing.jl/blob/master/CITATION.bib)
