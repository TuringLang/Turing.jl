# Getting Started

## Installation

To use Turing, you need install Julia first and then install Turing.

### Install Julia

You will need Julia 1.0, which you can get from [the official Julia website](http://julialang.org/downloads/).

It provides three options for users:

1. A command line version [Julia/downloads](http://julialang.org/downloads/) (**recommended**)
2. A community maintained IDE [Juno](http://www.junolab.org/)
3. [JuliaBox.com](https://www.juliabox.com/) - a Jupyter notebook in the browser

For command line version, we recommend that you install a version downloaded from Julia's [official website](http://julialang.org/downloads/), as Turing may not work correctly with Julia provided by other sources (e.g. Turing does not work with Julia installed via apt-get due to missing header files).

Juno also needs the command line version installed. This IDE is recommended for heavy users who require features like debugging, quick documentation check, etc.

JuliaBox provides a pre-installed Jupyter notebook for Julia. You can take a shot at Turing without installing Julia on your machine in few seconds.

### Install Turing.jl

Turing is an officially registered Julia package, so the following will install a stable version of Turing while inside Julia's package manager (press `]` from the REPL):

```julia
add Turing
```

[**Recommended**] If you want to use the latest version of Turing with some experimental features, you can try the following instead:

```julia
add Turing#master
test Turing
```

If all tests pass, you're ready to start using Turing.

## Basics

### Introduction

A probabilistic program is Julia code wrapped in a `@model` macro. It can use arbitrary Julia code, but to ensure correctness of inference it should not have external effects or modify global state. Stack-allocated variables are safe, but mutable heap-allocated objects may lead to subtle bugs when using task copying. To help avoid those we provide a Turing-safe datatype `TArray` that can be used to create mutable arrays in Turing programs.

For probabilistic effects, Turing programs should use the `~` notation:

`x ~ distr` where `x` is a symbol and `distr` is a distribution. If `x` is undefined in the model function, inside the probabilistic program, this puts a random variable named `x`, distributed according to `distr`, in the current scope. `distr` can be a value of any type that implements `rand(distr)`, which samples a value from the distribution `distr`. If `x` is defined, this is used for conditioning in a style similar to Anglican (another PPL). Here `x` should be a value that is observed to have been drawn from the distribution `distr`. The likelihood is computed using `logpdf(distr,y)` and should always be positive to ensure correctness of inference algorithms. The observe statements should be arranged so that every possible run traverses all of them in exactly the same order. This is equivalent to demanding that they are not placed inside stochastic control flow.

Available inference methods include  Importance Sampling (IS), Sequential Monte Carlo (SMC), Particle Gibbs (PG), Hamiltonian Monte Carlo (HMC), Hamiltonian Monte Carlo with Dual Averaging (HMCDA) and The No-U-Turn Sampler (NUTS).

### Simple Gaussian demo

Below is a simple Gaussian demo illustrate the basic usage of Turing.jl

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

Note: for the interests of sanity check, some analytical results on the expectation of samples of this model are E[s] = 49/24 and E[m] = 7/6.

Inference methods are functions which take the probabilistic program as one of the arguments.

```julia
#  Run sampler, collect results
c1 = sample(gdemo([1.5, 2]), SMC(1000))
c2 = sample(gdemo([1.5, 2]), PG(10,1000))
c3 = sample(gdemo([1.5, 2]), HMC(1000, 0.1, 5))
c4 = sample(gdemo([1.5, 2]), Gibbs(1000, PG(10, 2, :m), HMC(2, 0.1, 5, :s)))
c5 = sample(gdemo([1.5, 2]), HMCDA(1000, 0.15, 0.65))
c6 = sample(gdemo([1.5, 2]), NUTS(1000,  0.65))

# Summarise results
Mamba.describe(c3)

# Plot results
p = Mamba.plot(c3)
Mamba.draw(p, fmt=:pdf, filename="gdemo-plot.pdf")
```

The arguments for each sampler are

* SMC: number of particles
* PG: number of particles, number of iterations
* HMC: number of samples, leapfrog step size, leapfrog step numbers
* Gibbs: number of samples, component sampler 1, component sampler 2, ...
* HMCDA: number of samples, total leapfrog length, target accept ratio
* NUTS: number of samples, target accept ratio

For detailed information please check Turing.jl's [APIs](https://github.com/yebai/Turing.jl/wiki/APIs).

### Modelling syntax explained

Models are wrapped by `@model` with a normal function definition syntax, i.e.

```julia
@model model_name(arg_1, arg_2) = begin
  ...
end
```

This syntax defines a model which can take data as input to generate a posterior evaluator. The data can be provided either using the same function signature defined, or by using a dictionary containing each argument and its value as pairs, i.e.

```julia
model_func = model_name(1, 2)
model_func = model_name(Dict(:arg_1=>1, :arg_2=>2)
```

This posterior evaluator can then be called by a sampler to run inference, i.e.

```julia
chn = sample(model_func, HMC(..)) # do inference by sampling using HMC
```

The return chain contains samples of the variables in the model, one can use them do inference, e.g.

```julia
var_1 = mean(chn[:var_1]) # taking the mean of a variable named var_1
```

Note that the key should be a symbol. For this reason, in case of fetching `x[1]` one need to do `chn[Symbol(:x[1])`. Turing.jl provides a macro to work around this expression `chn[sym"x[1]"]`.

## Beyond basics

### Composition sampling using Gibbs

Turing.jl provides a Gibbs interface to combine different samplers. For example, one can combine a HMC sampler with a PG sampler to run inference for different parameters in a single model as below.

```julia
@model simple_choice(xs) = begin
  p ~ Beta(2, 2)
  z ~ Categorical(p)
  for x = xs
    if z == 1
      x ~ Normal(0, 1)
    else
      x ~ Normal(2, 1)
    end
  end
end

simple_choice_f = simple_choice([1.5, 2.0, 0.3])

chn = sample(simple_choice_f, Gibbs(1000, HMC(1,0.2,3,:p), PG(20,1,:z))
```

For details of composition sampling in Turing.jl, please check the corresponding [paper](http://xuk.ai/assets/aistats2018-turing.pdf).

### Working with Mamba.jl

Turing.jl wraps its samples using `Mamba.Chain` so that all the functions working for `Mamba.Chain` can be re-used in Turing.jl. Two typical functions are `Mamba.describe` and `Mamba.plot`, which can be used as follow for an obtained chain `chn`.

```julia
using Mamba: describe, plot

describe(chn) # gives statistics of the samples
plot(chn) # lots statistics of the samples
```

There are a plenty of functions which are useful in Mamaba.jl, e.g. those for convergence diagnostics at [here](http://mambajl.readthedocs.io/en/latest/tutorial.html#convergence-diagnostics).

### Changing default settings

Some of Turing.jl's default settings can be changed for better usage.

#### AD chunk size

Turing.jl uses ForwardDiff.jl for automatic differentiation, which uses the forward-mode chunk-wise AD. The chunk size can be manually set by `setchunksize(new_chunk_size)`, or alternatively, use an auto-tuning helper function `auto_tune_chunk_size!(mf::Function, rep_num=10)` which will do simple profile of using different chunk size and choose the best one. Here `mf` is the model function, e.g. `gdemo([1.5, 2])` and `rep_num` is the number of repetition for profiling.

#### AD backend

Since [#428](https://github.com/yebai/Turing.jl/pull/428), Turing.jl supports ReverseDiff.jl as backend. To switch between ForwardDiff.jl and ReverseDiff.jl, one can call function `setadbackend(backend_sym)`, where `backend_sym` can be `:forward_diff` or `:reverse_diff`.

#### Progress meter

Turing.jl uses ProgressMeter.jl to show the progress of sampling, which may lead to slow down of inference or even cause bugs in some IDEs due to I/O. This can be turned on or off by `turnprogress(true)` and `turnprogress(false)`, of which the former is set as default.
