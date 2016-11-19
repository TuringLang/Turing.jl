# Turing.jl

Turing is a Julia library for (_universal_) probabilistic programming.
It was originally created and is now managed by Hong Ge. 
The full list of [contributors](https://github.com/yebai/Turing.jl/graphs/contributors) is [Hong Ge](http://mlg.eng.cam.ac.uk/hong/), [Adam Scibior](http://mlg.eng.cam.ac.uk/?portfolio=adam-scibior), [Matej Balog](http://mlg.eng.cam.ac.uk/?portfolio=matej-balog), [Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/), [Kai Xu](http://mlg.eng.cam.ac.uk/?portfolio=kai-xu), [Emma Smith](https://github.com/evsmithx). Turing is an open source project so if you feel you have some relevant skills and are interested in contributing then please do get in touch.

[![Build Status](https://travis-ci.org/yebai/Turing.jl.svg?branch=master)](https://travis-ci.org/yebai/Turing.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/fvgi21998e1tfx0d/branch/master?svg=true)](https://ci.appveyor.com/project/yebai/turing-jl/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/yebai/Turing.jl/badge.svg?branch=master)](https://coveralls.io/github/yebai/Turing.jl?branch=master)
[![Turing](http://pkg.julialang.org/badges/Turing_0.4.svg)](http://pkg.julialang.org/?pkg=Turing)
[![Gitter](https://badges.gitter.im/gitterHQ/gitter.svg)](https://gitter.im/Turing-jl/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Documentation Status](https://readthedocs.org/projects/turingjl/badge/?version=latest)](http://turingjl.readthedocs.io/?badge=latest)

# Installation

To use Turing, you need install Julia first and then install Turing.

## 1) Install Julia

You will need Julia 0.5 (or 0.4; but 0.5 is recommended), which you can get from [the official Julia website](http://julialang.org/downloads/).

It provides three options for users

1. A command line version [Julia/downloads](http://julialang.org/downloads/)
2. A community maintained IDE [Juno](http://www.junolab.org/)
3. [JuliaBox.com](https://www.juliabox.com/) - a Jupyter notebook in the browser

For command line version, we recommend that you install a version downloaded from Julia's [official website](http://julialang.org/downloads/), as Turing may not work correctly with Julia provided by other sources (e.g. Turing does not work with Julia installed via apt-get due to missing header files).

Juno also needs the command line version installed. This IDE is recommended for heavy users who require features like debugging, quick documentation check, etc.

JuliaBox provides a pre-installed Jupyter notebook for Julia. You can take a shot at Turing without installing Julia on your machine in few seconds.

## 2) Install Turing

Turing is an officially registered Julia package, so the following should install a stable version of Turing:

```julia
Pkg.update()
Pkg.add("Turing")
Pkg.build("Turing")
Pkg.test("Turing")
```

If you want to use the latest version of Turing with some experimental features, you can try the following instead:

```julia
Pkg.update()
Pkg.clone("Turing")
Pkg.build("Turing")
Pkg.test("Turing")
```

If all tests pass, you're ready to start using Turing.

## 3) Simple Example

A Turing probabilistic program is just a normal Julia program, wrapped in a `@model` macro, that uses some of the special macros illustrated below. Available inference methods include  Importance Sampling (IS), Sequential Monte Carlo (SMC), Particle Gibbs (PG), Hamiltonian Monte Carlo (HMC).

```julia
# Define a simple Normal model with unknown mean and variance.
@model gaussdemo begin
  @assume s ~ InverseGamma(2,3)
  @assume m ~ Normal(0,sqrt(s))
  @observe 1.5 ~ Normal(m, sqrt(s))
  @observe 2.0 ~ Normal(m, sqrt(s))
  @predict s m
end
```

Inference methods are functions which take the probabilistic program as one of the arguments.

```julia
#  Run sampler, collect results
chain = sample(gaussdemo, SMC(500))
chain = sample(gaussdemo, PG(10,500))
chain = sample(gaussdemo, HMC(1000, 0.1, 5))
```

The arguments for each sampler are

* SMC: number of particles
* PG: number of praticles, number of iterations
* HMC: number of samples, leapfrog step size, leapfrog step numbers

# Citing Turing

To cite Turing, please refer to the technical report. Sample BibTeX entry is given below:

```
@ARTICLE{Turing2016,
    author = {Ge, Hong and {\'S}cibior, Adam and Xu, Kai and Ghahramani, Zoubin},
    title = "{Turing: A fast imperative probabilistic programming language.}",
    year = 2016,
    month = jun
}
```
