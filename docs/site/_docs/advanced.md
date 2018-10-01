---
title: Advanced Usage
permalink: /docs/advanced/
toc: true
toc_sticky: true
---

<a id='How-to-Define-a-Customized-Distribution-1'></a>

## How to Define a Customized Distribution


Turing.jl supports the use of distributions from the Distributions.jl package. By extension it also supports the use of customized distributions, by defining them as subtypes of `Distribution` type of the Distributions.jl package, as well as corresponding functions.


Below shows a workflow of how to define a customized distribution, using a flat prior as a simple example.


<a id='.-Define-the-Distribution-Type-1'></a>

### 1. Define the Distribution Type


First, define a type of the distribution, as a subtype of a corresponding distribution type in the Distributions.jl package.


```julia
immutable Flat <: ContinuousUnivariateDistribution
end
```


<a id='.-Implement-Sampling-and-Evaluation-of-the-log-pdf-1'></a>

### 2. Implement Sampling and Evaluation of the log-pdf


Second, define `rand()` and `logpdf()`, which will be used to run the model.


```julia
Distributions.rand(d::Flat) = rand()
Distributions.logpdf{T<:Real}(d::Flat, x::T) = zero(x)
```


<a id='.-Define-Helper-Functions-1'></a>

### 3. Define Helper Functions


In most cases, it may be required to define helper functions, such as the `minimum`, `maximum`, `rand`, and `logpdf` functions, among others.


<a id='.1-Domain-Transformation-1'></a>

#### 3.1 Domain Transformation


Some helper functions are necessary for domain transformation. For univariate distributions, the necessary ones to implement are `minimum()` and `maximum()`.


```julia
Distributions.minimum(d::Flat) = -Inf
Distributions.maximum(d::Flat) = +Inf
```


Functions for domain transformation which may be required by multivariate or matrix-variate distributions are `size(d)`, `link(d, x)` and `invlink(d, x)`. Please see Turing's [`transform.jl`](https://github.com/TuringLang/Turing.jl/blob/master/src/utilities/transform.jl) for examples.


<a id='.2-Vectorization-Support-1'></a>

#### 3.2 Vectorization Support


The vectorization syntax follows `rv ~ [distribution]`, which requires `rand()` and `logpdf()` to be called on multiple data points at once. An appropriate implementation for `Flat` are shown below.


```julia
Distributions.rand(d::Flat, n::Int) = Vector([rand() for _ = 1:n])
Distributions.logpdf{T<:Real}(d::Flat, x::Vector{T}) = zero(x)
```


<a id='Avoid-Using-the-@model-Macro-1'></a>

## Avoid Using the `@model` Macro


When integrating Turing.jl with other libraries, it can be necessary to avoid using the `@model` macro. To achieve this, one needs to understand the `@model` macro, which works as a closure and generates an amended function by


1. assigning the arguments to corresponding local variables;
2. adding two keyword arguments `vi=VarInfo()` and `sampler=nothing` to the scope; and
3. forcing the function to return `vi`.


Thus by doing these three steps manually, one can get rid of the `@model` macro. Taking the `gdemo` model as an example, the two code sections below (macro and macro-free) are equivalent.


```julia
@model gdemo(x, y) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt(s))
    x ~ Normal(m, sqrt(s))
    x ~ Normal(m, sqrt(s))
    return s, m
end

mf = gdemo(1.5, 2.0)
sample(mf, HMC(1000, 0.1, 5))
```


```julia
# Force Turing.jl to initialize its compiler
mf(vi, sampler; x=[1.5, 2.0]) = begin
  s = Turing.assume(sampler,
                    InverseGamma(2, 3),
                    Turing.VarName(vi, [:c_s, :s], ""),
                    vi)
  m = Turing.assume(sampler,
                    Normal(0,sqrt(s)),
                    Turing.VarName(vi, [:c_m, :m], ""),
                    vi)
  for i = 1:2
    Turing.observe(sampler,
                   Normal(m, sqrt(s)),
                   x[i],
                   vi)
  end
  vi
end
mf() = mf(Turing.VarInfo(), nothing)

sample(mf, HMC(1000, 0.1, 5))
```


Note that the use of `~` must be removed due to the fact that in Julia 0.6, `~` is no longer a macro. For this reason, Turing.jl parses `~` within the `@model` macro to allow for this intuitive notation.


<a id='Task-Copying-1'></a>

## Task Copying


Turing [copies](https://github.com/JuliaLang/julia/issues/4085) Julia tasks to deliver efficient inference algorithms, but it also provides alternative slower implementation as a fallback. Task copying is enabled by default. Task copying requires building a small C program, which should be done automatically on Linux and Mac systems that have GCC and Make installed.

