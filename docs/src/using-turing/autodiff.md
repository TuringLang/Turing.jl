---
title: Automatic Differentiation

---

# Automatic Differentiation

## Switching AD Modes


Turing supports four packages of automatic differentiation (AD) in the back end during sampling. The default AD backend is [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) for forward-mode AD. Three reverse-mode AD backends are also supported, namely [Tracker](https://github.com/FluxML/Tracker.jl), [Zygote](https://github.com/FluxML/Zygote.jl) and [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl). `Zygote` and `ReverseDiff` are supported optionally if explicitly loaded by the user with `using Zygote` or `using ReverseDiff` next to `using Turing`.

To switch between the different AD backends, one can call function `Turing.setadbackend(backend_sym)`, where `backend_sym` can be `:forwarddiff` (`ForwardDiff`), `:tracker` (`Tracker`), `:zygote` (`Zygote`) or `:reversediff` (`ReverseDiff.jl`). When using `ReverseDiff`, to compile the tape only once and cache it for later use, the user needs to load [Memoization.jl](https://github.com/marius311/Memoization.jl) first with `using Memoization` then call `Turing.setrdcache(true)`. However, note that the use of caching in certain types of models can lead to incorrect results and/or errors. Models for which the compiled tape can be safely cached are models with fixed size loops and no run-time if statements. Compile-time if statements are fine. To empty the cache, you can call `Turing.emptyrdcache()`.


## Compositional Sampling with Differing AD Modes


Turing supports intermixed automatic differentiation methods for different variable spaces. The snippet below shows using `ForwardDiff` to sample the mean (`m`) parameter, and using the Tracker-based `TrackerAD` autodiff for the variance (`s`) parameter:


```julia
using Turing

# Define a simple Normal model with unknown mean and variance.
@model function gdemo(x, y)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end

# Sample using Gibbs and varying autodiff backends.
c = sample(
	gdemo(1.5, 2),
  	Gibbs(
    	HMC{Turing.ForwardDiffAD{1}}(0.1, 5, :m),
        HMC{Turing.TrackerAD}(0.1, 5, :s)
    ),
    1000,
)
```


Generally, `TrackerAD` is faster when sampling from variables of high dimensionality (greater than 20) and `ForwardDiffAD` is more efficient for lower-dimension variables. This functionality allows those who are performance sensistive to fine tune their automatic differentiation for their specific models.


If the differentation method is not specified in this way, Turing will default to using whatever the global AD backend is. Currently, this defaults to `ForwardDiff`.

