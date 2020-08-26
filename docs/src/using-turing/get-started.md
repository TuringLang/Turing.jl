---
title: Getting Started
---

# Getting Started


## Installation

To use Turing, you need to install Julia first and then install Turing.


### Install Julia

You will need to install Julia 1.3 or greater, which you can get from [the official Julia website](http://julialang.org/downloads/).


### Install Turing.jl

Turing is an officially registered Julia package, so you can install a stable version of Turing by running the following in the Julia REPL:

```julia
julia> ] add Turing
```

You can check if all tests pass by running

```julia
julia> ] test Turing
```

### Example

Here's a simple example showing the package in action:


```julia
using Turing
using StatsPlots

# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x, y)
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
end

#  Run sampler, collect results
chn = sample(gdemo(1.5, 2), HMC(0.1, 5), 1000)

# Summarise results
describe(chn)

# Plot and save results
p = plot(chn)
savefig("gdemo-plot.png")
```
