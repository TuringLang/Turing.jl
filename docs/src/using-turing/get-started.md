---
title: Getting Started
---

# Getting Started


## Installation

To use Turing, you need to install Julia first and then install Turing.


### Install Julia

You will need to install Julia 1.0 or greater, which you can get from [the official Julia website](http://julialang.org/downloads/).


### Install Turing.jl

Turing is an officially registered Julia package, so the following will install a stable version of Turing while inside Julia's package manager (press `]` from the REPL):


```julia
add Turing
```


If you want to use the latest version of Turing with some experimental features, you can try the following instead:


```julia
add Turing#master
test Turing
```


If all tests pass, you're ready to start using Turing.


### Example

Here's a simple example showing the package in action:


```julia
using Turing
using StatsPlots

# Define a simple Normal model with unknown mean and variance.
@model gdemo(x, y) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
end

#  Run sampler, collect results
chn = sample(gdemo(1.5, 2), HMC(1000, 0.1, 5))

# Summarise results (currently requires the master branch from MCMCChains)
describe(chn)

# Plot and save results
p = plot(chn)
savefig("gdemo-plot.png")
```

