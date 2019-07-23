---
title: Using DynamicHMC
---

# Using DynamicHMC

Turing supports the use of [DynamicHMC](https://github.com/tpapp/DynamicHMC.jl) as a sampler through the use of the `DynamicNUTS` function. This is a [faster](https://github.com/TuringLang/Turing.jl/issues/559) version of Turing's native `NUTS` implementation.


`DynamicNUTS` is not appropriate for use in compositional inference. If you intend to use [Gibbs](http://turing.ml/docs/library/#-turinggibbs--type) sampling, you must use Turing's native `NUTS` function.


To use the `DynamicNUTS` function, you must import the `DynamicHMC` package as well as Turing. Turing does not formally require `DynamicHMC` but will include additional functionality if both packages are present.


Here is a brief example of how to apply `DynamicNUTS`:


```julia
# Import Turing and DynamicHMC.
using LogDensityProblems, DynamicHMC, Turing

# Model definition.
@model gdemo(x, y) = begin
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))
  x ~ Normal(m, sqrt(s))
  y ~ Normal(m, sqrt(s))
end

# Pull 2,000 samples using DynamicNUTS.
chn = sample(gdemo(1.5, 2.0), DynamicNUTS(2000))
```

