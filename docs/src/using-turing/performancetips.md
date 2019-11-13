---
title: Performance Tips
---

# Performance Tips

This section briefly summarieses a few common techniques to ensure good performance when using Turing.
We refer to [julialang.org](https://docs.julialang.org/en/v1/manual/performance-tips/index.html) for general techniques to ensure good performance of Julia programs.

## Use multivariate distributions

It is generally preferable to use mutlivariate distributions if possible.

The following example:

```julia
@model gmodel(x) = begin
    m ~ Normal()
    for i = 1:length(x)
        x[i] ~ Normal(m, 0.2)
    end
end
```

can be directly expressed more efficiently using a simple transformation:

```julia
@model gmodel(x) = begin
    m ~ Normal()
    x ~ MvNormal(fill(m, length(x)), 0.2)
end
```

### Make your model type-stable

For efficient gradient-based inference, e.g. using HMC, NUTS or ADVI, it is important to ensure the model is type-stable.
We refer to [julialang.org](https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Write-"type-stable"-functions-1) for a general discussion on type-stability.

The following example:

```julia
@model tmodel(x, y) = begin
    p,n = size(x)
    params = Vector{Real}(undef, n)
    for i = 1:n
        params[i] ~ Truncated(Normal(), 0, Inf)
    end

    a = x * params
    y ~ MvNormal(a, 1.0)
end
```

can be transformed into the th following type-stable representation:

```julia
@model tmodel(x, y, ::Type{T}=Vector{Float64}) where {T} = begin
    p,n = size(x)
    params = T(undef, n)
    for i = 1:n
        params[i] ~ Truncated(Normal(), 0, Inf)
    end

    a = x * params
    y ~ MvNormal(a, 1.0)
end
```

