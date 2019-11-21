---
title: Performance Tips
---

# Performance Tips

This section briefly summarises a few common techniques to ensure good performance when using Turing.
We refer to [julialang.org](https://docs.julialang.org/en/v1/manual/performance-tips/index.html) for general techniques to ensure good performance of Julia programs.

## Use multivariate distributions

It is generally preferable to use multivariate distributions if possible.

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
## Choose your AD backend
Turing currently provides support for two different automatic differentiation (AD) backends. 
Generally, try to use `:forward_diff` for models with few parameters and `:reverse_diff` for models with large parameter vectors or linear algebra operations. See [Automatic Differentiation](autodiff) for details.

## Special care for `reverse_diff`

In case of `reverse_diff` it is necessary to avoid loops for now.
This is mainly due to the reverse-mode AD backend `Tracker` which is inefficient for such cases.
Therefore, it is often recommended to write a [custom distribution](advanced) which implements a multivariate version of the prior distribution.

## Make your model type-stable

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
can be transformed into the following type-stable representation:

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

Note that you can use `@code_warntype` to find type instabilities in your model definition.

For example consider the following simple program

```julia
@model tmodel(x) = begin
	p = Vector{Real}(undef, 1); 
	p[1] ~ Normal()
	p = p .+ 1
	x ~ Normal(p[1])
end
```
we can use

```julia
m = tmodel(1.0);
varinfo = Turing.VarInfo(model);
spl = Turing.SampleFromPrior();

@code_warntype model.f(varinfo, spl, model);
```
to inspect the type instabilities in the model.