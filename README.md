# Turing.jl

[![Build Status](https://travis-ci.org/yebai/Turing.jl.svg?branch=master)](https://travis-ci.org/yebai/Turing.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/fvgi21998e1tfx0d/branch/master?svg=true)](https://ci.appveyor.com/project/yebai/turing-jl/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/yebai/Turing.jl/badge.svg?branch=master)](https://coveralls.io/github/yebai/Turing.jl?branch=master)
[![Turing](http://pkg.julialang.org/badges/Turing_0.5.svg)](http://pkg.julialang.org/?pkg=Turing)
[![Gitter](https://badges.gitter.im/gitterHQ/gitter.svg)](https://gitter.im/Turing-jl/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Wiki Status](https://img.shields.io/badge/wiki-v0.3.0-green.svg)](https://github.com/yebai/Turing.jl/wiki)

**Turing.jl** is a Julia library for (_universal_) probabilistic programming. Current features include:

- Universal probabilistic programming with an intuitive modelling interface
- Hamiltonian Monte Carlo (HMC) sampling for differentiable posterior distributions
- Particle MCMC sampling for complex posterior distributions involving discrete variables and stochastic control flows
- Compositional MCMC sampling that combines particle MCMC and HMC

## Turing.jl wiki

Please visit [Turing.jl wiki](https://github.com/yebai/Turing.jl/wiki) for tutorials (e.g. [Get started](https://github.com/yebai/Turing.jl/wiki/Get-started)) and other topics (e.g. [Advanced usages](https://github.com/yebai/Turing.jl/wiki/Advanced-usages)).

## Contributing
Turing is an open source project so if you feel you have some relevant skills and are interested in contributing then please do get in touch. You can contribute by opening issues on Github or implementing things yourself and making a pull request. We would also appreciate example models written using Truing to add to examples.

## Contributors

Turing was originally created and is now managed by Hong Ge. Current and past Turing team members include [Hong Ge](http://mlg.eng.cam.ac.uk/hong/), [Adam Scibior](http://mlg.eng.cam.ac.uk/?portfolio=adam-scibior), [Matej Balog](http://mlg.eng.cam.ac.uk/?portfolio=matej-balog), [Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/), [Kai Xu](http://mlg.eng.cam.ac.uk/?portfolio=kai-xu), [Emma Smith](https://github.com/evsmithx). 
You can see the full list of on Github: https://github.com/yebai/Turing.jl/graphs/contributors. Thanks for the important additions, fixes and comments.

## Relevant papers

1. Ghahramani, Zoubin. “Probabilistic machine learning and artificial intelligence.” Nature 521, no. 7553 (2015): 452-459. ([pdf](http://www.nature.com/nature/journal/v521/n7553/full/nature14541.html))
2. Ge, Hong and Ścibior, Adam and Xu, Kai and Ghahramani, Zoubin. “Turing: A fast imperative probabilistic programming language.”

## Citing Turing.jl ##

To cite Turing, please refer to the technical report. Sample BibTeX entry is given below:

```
@ARTICLE{Turing2016,
    author = {Ge, Hong and {\'S}cibior, Adam and Xu, Kai and Ghahramani, Zoubin},
    title = "{Turing: A fast imperative probabilistic programming language.}",
    year = 2016,
    month = jun
}
```

