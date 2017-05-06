# Turing

[![Build Status](https://travis-ci.org/yebai/Turing.jl.svg?branch=master)](https://travis-ci.org/yebai/Turing.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/fvgi21998e1tfx0d/branch/master?svg=true)](https://ci.appveyor.com/project/yebai/turing-jl/branch/master)
[![Coverage Status](https://coveralls.io/repos/github/yebai/Turing.jl/badge.svg?branch=master)](https://coveralls.io/github/yebai/Turing.jl?branch=master)
[![Turing](http://pkg.julialang.org/badges/Turing_0.5.svg)](http://pkg.julialang.org/?pkg=Turing)
[![Gitter](https://badges.gitter.im/gitterHQ/gitter.svg)](https://gitter.im/Turing-jl/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![Documentation Status](https://readthedocs.org/projects/turingjl/badge/?version=latest)](http://turingjl.readthedocs.io/?badge=latest)

**Turing** is a Julia library for (_universal_) probabilistic programming. Current features include:

- Universal probabilistic programming with an intuitive modelling interface
- Hamiltonian Monte Carlo (HMC) sampling for differentiable posterior distributions
- Particle MCMC sampling for complex posterior distributions involving discrete variables and stochastic control flows
- Compositional MCMC sampling that combines particle MCMC and HMC

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
Pkg.add("Turing")
```

If you want to use the latest version of Turing with some experimental features, you can try the following instead:

```julia
Pkg.update()
Pkg.clone("Turing")
Pkg.build("Turing")
Pkg.test("Turing")
```

If all tests pass, you're ready to start using Turing.

## 3) Usage

A Turing probabilistic program is just a normal Julia program, wrapped in a `@model` macro, that uses some of the special macros illustrated below. Available inference methods include  Importance Sampling (IS), Sequential Monte Carlo (SMC), Particle Gibbs (PG), Hamiltonian Monte Carlo (HMC).

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end
```

Inference methods are functions which take the probabilistic program as one of the arguments.

```julia
#  Run sampler, collect results
c1 = sample(gdemo([1.5, 2]), SMC(1000))
c2 = sample(gdemo([1.5, 2]), PG(10,1000))
c3 = sample(gdemo([1.5, 2]), HMC(1000, 0.1, 5))
c4 = sample(gdemo([1.5, 2]), Gibbs(1000, PG(10, 2, :m), HMC(2, 0.1, 5, :s)))

# Summarise results
describe(c3)

# Plot results 
p = Turing.plot(c3)
Turing.draw(p, fmt=:pdf, filename="gdemo-plot.pdf")
```

The arguments for each sampler are

* SMC: number of particles
* PG: number of praticles, number of iterations
* HMC: number of samples, leapfrog step size, leapfrog step numbers
* Gibbs: number of samples, component sampler 1, component sampler 2, ...


## Contributing
Turing is an open source project so if you feel you have some relevant skills and are interested in contributing then please do get in touch. You can contribute by opening issues on Github or implementing things yourself and making a pull request. We would also appreciate example models written using Truing to add to examples.

## Contributors

Turing was originally created and is now managed by Hong Ge. Current and past Turing team members include [Hong Ge](http://mlg.eng.cam.ac.uk/hong/), [Adam Scibior](http://mlg.eng.cam.ac.uk/?portfolio=adam-scibior), [Matej Balog](http://mlg.eng.cam.ac.uk/?portfolio=matej-balog), [Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/), [Kai Xu](http://mlg.eng.cam.ac.uk/?portfolio=kai-xu), [Emma Smith](https://github.com/evsmithx). 
You can see the full list of on Github: https://github.com/yebai/Turing.jl/graphs/contributors. Thanks for the important additions, fixes and comments.


## Citing Turing ##

To cite Turing, please refer to the technical report. Sample BibTeX entry is given below:

```
@ARTICLE{Turing2016,
    author = {Ge, Hong and {\'S}cibior, Adam and Xu, Kai and Ghahramani, Zoubin},
    title = "{Turing: A fast imperative probabilistic programming language.}",
    year = 2016,
    month = jun
}
```

