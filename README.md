# Turing.jl

Turing is a Julia library for (universal) probabilistic programming. 
The full list of [contributors](https://github.com/yebai/Turing.jl/graphs/contributors) (in alphabetical order) is [Hong Ge](http://mlg.eng.cam.ac.uk/hong/), [Adam Scibior](http://mlg.eng.cam.ac.uk/?portfolio=adam-scibior), [Matej Balog](http://mlg.eng.cam.ac.uk/?portfolio=matej-balog), [Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/), [Kai Xu](http://mlg.eng.cam.ac.uk/?portfolio=kai-xu), [Emma Smith](https://github.com/evsmithx). Turing is an open source project so if you feel you have some relevant skills and are interested in contributing then please do contact us.

[![Build Status](https://travis-ci.org/yebai/Turing.jl.svg?branch=master)](https://travis-ci.org/yebai/Turing.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/fvgi21998e1tfx0d/branch/master?svg=true)](https://ci.appveyor.com/project/yebai/turing-jl/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/yebai/Turing.jl/badge.svg?branch=master)](https://coveralls.io/github/yebai/Turing.jl?branch=master)
[![Turing](http://pkg.julialang.org/badges/Turing_0.4.svg)](http://pkg.julialang.org/?pkg=Turing)
[![Gitter](https://badges.gitter.im/gitterHQ/gitter.svg)](https://gitter.im/Turing-jl/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Documentation Status](https://readthedocs.org/projects/turingjl/badge/?version=latest)](http://turingjl.readthedocs.io/?badge=latest)

## Installation

You will need Julia 0.5 (or 0.4; but 0.5 is recommended), which you can get from the official Julia [website](http://julialang.org/downloads/). 

Turing is an officially registered Julia package, so the following should work:

```julia
Pkg.update()
Pkg.add("Turing")
Pkg.test("Turing")
```

If Turing can not be located or you want to use the latest version of Turing, you can try the following instead:

```julia
Pkg.clone("https://github.com/yebai/Turing.jl")
Pkg.build("Turing")
Pkg.test("Turing")
```

If all tests pass, you're ready to start using Turing.

### Example

A Turing probabilistic program is just a normal Julia program, wrapped in a `@model` macro, that uses some of the special macros illustrated below. Available inference methods include  Importance Sampling (IS), Sequential Monte Carlo (SMC), Particle Gibbs (PG), Hamiltonian Monte Carlo (HMC).

```julia
@model gaussdemo begin
  # Define a simple Normal model with unknown mean and variance.
  @assume s ~ InverseGamma(2,3)
  @assume m ~ Normal(0,sqrt(s))
  @observe 1.5 ~ Normal(m, sqrt(s))
  @observe 2.0 ~ Normal(m, sqrt(s))
  @predict s m
end
```

Inference methods are functions which take the probabilistic program as one of the arguments.
```julia
#  run sampler, collect results
chain = sample(gaussdemo, SMC(500))
chain = sample(gaussdemo, PG(10,500))
chain = sample(gaussdemo, HMC(10,500))
```

### Relevant papers
1. Ghahramani, Zoubin. "Probabilistic machine learning and artificial intelligence." Nature 521, no. 7553 (2015): 452-459. ([pdf](http://www.nature.com/nature/journal/v521/n7553/full/nature14541.html))
2. Ge, Hong, Adam Scibior, and Zoubin Ghahramani "Turing: A fast imperative probabilistic programming language." (In submission).
