---
title: Using DynamicHMC
---

# Using DynamicHMC

Turing supports the use of [DynamicHMC](https://github.com/tpapp/DynamicHMC.jl) as a sampler through the `DynamicNUTS` function.

To use the `DynamicNUTS` function, you must import the `DynamicHMC` package as well as Turing. Turing does not formally require `DynamicHMC` but will include additional functionality if both packages are present.

Here is a brief example of how to apply `DynamicNUTS`:


```julia
# Import Turing and DynamicHMC.
using DynamicHMC, Turing

# Model definition.
@model function gdemo(x, y)
  s² ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s²))
  x ~ Normal(m, sqrt(s²))
  y ~ Normal(m, sqrt(s²))
end

# Pull 2,000 samples using DynamicNUTS.
chn = sample(gdemo(1.5, 2.0), DynamicNUTS(), 2000)
```
