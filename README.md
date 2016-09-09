# Turing.jl
[![Build Status](https://travis-ci.org/yebai/Turing.jl.svg?branch=development)](https://travis-ci.org/yebai/Turing.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/fvgi21998e1tfx0d/branch/development?svg=true)](https://ci.appveyor.com/project/yebai/turing-jl/branch/development)
[![Coverage Status](https://coveralls.io/repos/github/yebai/Turing.jl/badge.svg?branch=development)](https://coveralls.io/github/yebai/Turing.jl?branch=development)
[![Turing](http://pkg.julialang.org/badges/Turing_0.4.svg)](http://pkg.julialang.org/?pkg=Turing)


Turing is a Julia library for probabilistic programming. A Turing probabilistic program is just a normal Julia program, wrapped in a `@model` macro, that uses some of the special macros listed below. Available inference methods include  Importance Sampling, Sequential Monte Carlo, Particle Gibbs.

Authors: [Hong Ge](http://mlg.eng.cam.ac.uk/hong/), [Adam Scibior](http://mlg.eng.cam.ac.uk/?portfolio=adam-scibior), [Matej Balog](http://mlg.eng.cam.ac.uk/?portfolio=matej-balog), [Zoubin Ghahramani](http://mlg.eng.cam.ac.uk/zoubin/)

### Relevant papers
1. Ghahramani, Zoubin. "Probabilistic machine learning and artificial intelligence." Nature 521, no. 7553 (2015): 452-459. ([pdf](http://www.nature.com/nature/journal/v521/n7553/full/nature14541.html))
2. Ge, Hong, Adam Scibior, and Zoubin Ghahramani "Turing: rejuvenating probabilistic programming in Julia." (In submission).


### Example
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

## Installation

You will need Julia 0.4, which you can get from the official Julia [website](http://julialang.org/downloads/). We recommend that you install a pre-compiled package, as Turing may not work correctly with Julia built form source.

Turing is an officially registered Julia package, so the following should work:

```julia
Pkg.update()
Pkg.add("Turing")
Pkg.test("Turing")
```

If Turing can not be located, you can try the following instead:

```julia
Pkg.clone("https://github.com/yebai/Turing.jl")
Pkg.build("Turing")
Pkg.test("Turing")
```

If all tests pass, you're ready to start using Turing.

## Modelling API
A probabilistic program is Julia code wrapped in a `@model` macro. It can use arbitrary Julia code, but to ensure correctness of inference it should not have external effects or modify global state. Stack-allocated variables are safe, but mutable heap-allocated objects may lead to subtle bugs when using task copying. To help avoid those we provide a Turing-safe datatype `TArray` that can be used to create mutable arrays in Turing programs.

For probabilistic effects, Turing programs should use the following macros:

`@assume x ~ distr`
where `x` is a symbol and `distr` is a distribution. Inside the probabilistic program this puts a random variable named `x`, distributed according to `distr`, in the current scope. `distr` can be a value of any type that implements `rand(distr)`, which samples a value from the distribution `distr`.

`@observe y ~ distr`
This is used for conditioning in a style similar to Anglican. Here `y` should be a value that is observed to have been drawn from the distribution `distr`. The likelihood is computed using `pdf(distr,y)` and should always be positive to ensure correctness of inference algorithms. The observe statements should be arranged so that every possible run traverses all of them in exactly the same order. This is equivalent to demanding that they are not placed inside stochastic control flow.

`@predict x`
Registers the current value of `x` to be inspected in the results of inference.

## Inference API
Inference methods are functions which take the probabilistic program as one of the arguments.
```julia
#  run sampler, collect results
chain = sample(gaussdemo, SMC(500))
chain = sample(gaussdemo, PG(10,500))
```

## Task copying
Turing [copies](https://github.com/JuliaLang/julia/issues/4085) Julia tasks to deliver efficient inference algorithms, but it also provides alternative slower implementation as a fallback. Task copying is enabled by default. Task copying requires building a small C program, which should be done automatically on Linux and Mac systems that have GCC and Make installed.

## Development notes
Following GitHub guidelines, we have two main branches: master and development. We protect them with a review process to make sure at least two people approve the code before it is committed to either of them. For this reason, do not commit to either master or development directly.
Please use the following workflow instead.

### Reporting bugs
- branch from master
- write a test that exposes the bug
- create an issue describing the bug, referencing the new branch

### Bug fixes
- assign yourself to the relevant issue
- fix the bug on the dedicated branch
- see that the new test (and all the old ones) passes
- create a pull request
- when a pull request is accepted close the issue and propagate changes to development

### New features and performance enhancements
- create an issue describing the proposed change
- branch from development
- write tests for the new features
- implement the feature
- see that all tests pass
- create a pull request

### Merging development into master
- review changes from master
- create a pull request

## External contributions
Turing is an open-source project and we welcome any and all contributions from the community. If you have comments, questions, suggestions, or bug reports, please open an issue and let us know. If you want to contribute bug fixes or new features, please fork the repo and make a pull requrest. If you'd like to make a more substantial contribution to Turing, please get in touch so we can discuss the best way to proceed.
