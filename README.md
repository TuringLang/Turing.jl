# Turing.jl

[![Build Status](https://travis-ci.org/TuringLang/Turing.jl.svg?branch=master)](https://travis-ci.org/TuringLang/Turing.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/gp1xtxsc3971pwi6/branch/master?svg=true)](https://ci.appveyor.com/project/TuringLang/turing-jl/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/yebai/Turing.jl/badge.svg?branch=master)](https://coveralls.io/github/yebai/Turing.jl?branch=master)
[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](http://turing.ml/docs/)

**Turing.jl** is a Julia library for (_universal_) [probabilistic programming](https://en.wikipedia.org/wiki/Probabilistic_programming_language). Current features include:

- Universal probabilistic programming with an intuitive modelling interface
- Hamiltonian Monte Carlo (HMC) sampling for differentiable posterior distributions
- Particle MCMC sampling for complex posterior distributions involving discrete variables and stochastic control flows
- Gibbs sampling that combines particle MCMC,  HMC and many other MCMC algorithms

## Getting Started

Turing's home page, with links to everything you'll need to use Turing is:

http://turing.ml/docs/get-started/


## Contributing

Turing was originally created and is now managed by Hong Ge. Current and past Turing team members include [Hong Ge](http://mlg.eng.cam.ac.uk/hong/), [Adam Scibior](http://mlg.eng.cam.ac.uk/?portfolio=adam-scibior), [Matej Balog](http://mlg.eng.cam.ac.uk/?portfolio=matej-balog), [Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/), [Kai Xu](http://mlg.eng.cam.ac.uk/?portfolio=kai-xu), [Emma Smith](https://github.com/evsmithx), [Emile Mathieu](http://emilemathieu.fr), [Martin Trapp](http://martint.blog).
You can see the full list of on Github: https://github.com/TuringLang/Turing.jl/graphs/contributors.

Turing is an open source project so if you feel you have some relevant skills and are interested in contributing then please do get in touch. See the [Contributing](http://turing.ml/docs/contributing/) page for details on the process. You can contribute by opening issues on Github or implementing things yourself and making a pull request. We would also appreciate example models written using Turing.

## Slack

Join [our channel](https://julialang.slack.com/messages/turing/) (`#turing`) in the Julia Slack chat for help, discussion, or general communication with the Turing team. If you do not already have an invitation to Julia's Slack, you can get one by going [here](https://slackinvite.julialang.org/).

## Citing Turing.jl ##
If you use **Turing** for your own research, please consider citing the following publication: Hong Ge, Kai Xu, and Zoubin Ghahramani: **Turing: a language for flexible probabilistic inference.** AISTATS 2018 [pdf](http://proceedings.mlr.press/v84/ge18b.html) [bibtex](https://dblp.org/rec/bib2/conf/aistats/GeXG18.bib)
