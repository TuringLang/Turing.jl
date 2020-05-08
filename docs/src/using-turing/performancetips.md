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
Generally, try to use `:forwarddiff` for models with few parameters and `:reversediff`, `:tracker` or `:zygote` for models with large parameter vectors or linear algebra operations. See [Automatic Differentiation](autodiff) for details.


## Special care for `:tracker` and `:zygote`

In case of `:tracker` and `:zygote`, it is necessary to avoid loops for now.
This is mainly due to the reverse-mode AD backends `Tracker` and `Zygote` which are inefficient for such cases. `ReverseDiff` does better but vectorized operations will still perform better.

Avoiding loops can be done using `filldist(dist, N)` and `arraydist(dists)`. `filldist(dist, N)` creates a multivariate distribution that is composed of `N` identical and independent copies of the univariate distribution `dist` if `dist` is univariate, or it creates a matrix-variate distribution composed of `N` identical and idependent copies of the multivariate distribution `dist` if `dist` is multivariate. `filldist(dist, N, M)` can also be used to create a matrix-variate distribution from a univariate distribution `dist`.  `arraydist(dists)` is similar to `filldist` but it takes an array of distributions `dists` as input. Writing a [custom distribution](advanced) with a custom adjoint is another option to avoid loops.


## Make your model type-stable

For efficient gradient-based inference, e.g. using HMC, NUTS or ADVI, it is important to ensure the model is type-stable.
We refer to [julialang.org](https://docs.julialang.org/en/v1/manual/performance-tips/index.html#Write-"type-stable"-functions-1) for a general discussion on type-stability.

The following example:

```julia
@model tmodel(x, y) = begin
    p,n = size(x)
    params = Vector{Real}(undef, n)
    for i = 1:n
        params[i] ~ truncated(Normal(), 0, Inf)
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
        params[i] ~ truncated(Normal(), 0, Inf)
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
model = tmodel(1.0)
varinfo = Turing.VarInfo(model)
spl = Turing.SampleFromPrior()

@code_warntype model.f(model, varinfo, spl, Turing.DefaultContext())
```
to inspect the type instabilities in the model.


## Reuse Computations in Gibbs Sampling

Often when performing Gibbs sampling, one can save computational time by caching the output of expensive functions. The cached values can then be reused in future Gibbs sub-iterations which do not change the inputs to this expensive function. For example in the following model:
```julia
@model demo(x) = begin
    a ~ Gamma()
    b ~ Normal()
    c = function1(a)
    d = function2(b)
    x .~ Normal(c, d)
end
alg = Gibbs(MH(:a), MH(:b))
sample(demo(zeros(10)), alg, 1000)
```
when only updating `a` in a Gibbs sub-iteration, keeping `b` the same, the value of `d` doesn't change. And when only updating `b`, the value of `c` doesn't change. However, if `function1` and `function2` are expensive and are both run in every Gibbs sub-iteration, a lot of time would be spent computing values that we already computed before. Such a problem can be overcome using `Memoization.jl`. Memoizing a function lets us store and reuse the output of the function for every input it is called with. This has a slight time overhead but for expensive functions, the savings will be far greater. 

To use `Memoization.jl`, simply define memoized versions of `function1` and `function2` as such:
```julia
using Memoization

@memoize memoized_function1(args...) = function1(args...)
@memoize memoized_function2(args...) = function2(args...)
```
Then define the `Turing` model using the new functions as such:
```julia
@model demo(x) = begin
    a ~ Gamma()
    b ~ Normal()
    c = memoized_function1(a)
    d = memoized_function2(b)
    x .~ Normal(c, d)
end
```
