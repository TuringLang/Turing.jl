---
title: Automatic Differentiation

---

# Automatic Differentiation

## Switching AD Modes


Turing supports two types of automatic differentiation (AD) in the back end during sampling. The current default AD mode is [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl), but Turing also supports [Tracker](https://github.com/FluxML/Tracker.jl)-based differentation.


To switch between `ForwardDiff` and `Tracker`, one can call function `Turing.setadbackend(backend_sym)`, where `backend_sym` can be `:forward_diff` or `:reverse_diff`.


## Compositional Sampling with Differing AD Modes


Turing supports intermixed automatic differentiation methods for different variable spaces. The snippet below shows using `ForwardDiff` to sample the mean (`m`) parameter, and using the Tracker-based `TrackerAD` autodiff for the variance (`s`) parameter:


```julia
using Turing

# Define a simple Normal model with unknown mean and variance.
@model gdemo(x, y) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end

# Sample using Gibbs and varying autodiff backends.
c = sample(gdemo(1.5, 2),
  Gibbs(1000,
    HMC{Turing.ForwardDiffAD{1}}(2, 0.1, 5, :m),
    HMC{Turing.TrackerAD}(2, 0.1, 5, :s)))
```


Generally, `TrackerAD` is faster when sampling from variables of high dimensionality (greater than 20) and `ForwardDiffAD` is more efficient for lower-dimension variables. This functionality allows those who are performance sensistive to fine tune their automatic differentiation for their specific models.


If the differentation method is not specified in this way, Turing will default to using whatever the global AD backend is. Currently, this defaults to `ForwardDiff`.

