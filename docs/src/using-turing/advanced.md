---
title: Advanced Usage
---

# Advanced Usage

## How to Define a Customized Distribution


Turing.jl supports the use of distributions from the Distributions.jl package. By extension it also supports the use of customized distributions, by defining them as subtypes of `Distribution` type of the Distributions.jl package, as well as corresponding functions.


Below shows a workflow of how to define a customized distribution, using our own implementation of a simple `Uniform` distribution as a simple example.


### 1. Define the Distribution Type


First, define a type of the distribution, as a subtype of a corresponding distribution type in the Distributions.jl package.


```julia
struct CustomUniform <: ContinuousUnivariateDistribution
end
```

### 2. Implement Sampling and Evaluation of the log-pdf


Second, define `rand` and `logpdf`, which will be used to run the model.


```julia
Distributions.rand(rng::AbstractRNG, d::CustomUniform) = rand(rng) # sample in [0, 1]
Distributions.logpdf(d::CustomUniform, x::Real) = zero(x)          # p(x) = 1 → logp(x) = 0
```

### 3. Define Helper Functions


In most cases, it may be required to define some helper functions.

#### 3.1 Domain Transformation

Certain samplers, such as `HMC`, require the domain of the priors to be unbounded. Therefore, to use our `CustomUniform` as a prior in a model we also need to define how to transform samples from `[0, 1]` to `ℝ`. To do this, we simply need to define the corresponding `Bijector` from `Bijectors.jl`, which is what `Turing.jl` uses internally to deal with constrained distributions.

To transform from `[0, 1]` to `ℝ` we can use the `Logit` bijector:

```julia
Bijectors.bijector(d::CustomUniform) = Logit(0., 1.)
```

You'd do the exact same thing for `ContinuousMultivariateDistribution` and `ContinuousMatrixDistribution`. For example, `Wishart` defines a distribution over positive-definite matrices and so `bijector` returns a `PDBijector` when called with a `Wishart` distribution as an argument. For discrete distributions, there is no need to define a bijector; the `Identity` bijector is used by default.

Alternatively, for `UnivariateDistribution` we can define the `minimum` and `maximum` of the distribution

```julia
Distributions.minimum(d::CustomUniform) = 0.
Distributions.maximum(d::CustomUniform) = 1.
```

and `Bijectors.jl` will return a default `Bijector` called `TruncatedBijector` which makes use of `minimum` and `maximum` derive the correct transformation.

Internally, Turing basically does the following when it needs to convert a constrained distribution to an unconstrained distribution, e.g. when sampling using `HMC`:
```julia
b = bijector(dist)
transformed_dist = transformed(dist, b) # results in distribution with transformed support + correction for logpdf
```
and then we can call `rand` and `logpdf` as usual, where
- `rand(transformed_dist)` returns a sample in the unconstrained space, and
- `logpdf(transformed_dist, y)` returns the log density of the original distribution, but with `y` living in the unconstrained space.

To read more about Bijectors.jl, check out [the project README](https://github.com/TuringLang/Bijectors.jl).

#### 3.2 Vectorization Support


The vectorization syntax follows `rv ~ [distribution]`, which requires `rand` and `logpdf` to be called on multiple data points at once. An appropriate implementation for `Flat` is shown below.


```julia
Distributions.logpdf(d::Flat, x::AbstractVector{<:Real}) = zero(x)
```


## Model Internals


The `@model` macro accepts a function definition and generates a `Turing.Model` struct for use by the sampler. Models can be constructed by hand without the use of a macro. Taking the `gdemo` model as an example, the two code sections below (macro and macro-free) are equivalent.


```julia
using Turing

@model gdemo(x) = begin
  # Set priors.
  s ~ InverseGamma(2, 3)
  m ~ Normal(0, sqrt(s))

  # Observe each value of x.
  @. x ~ Normal(m, sqrt(s))
end

sample(gdemo([1.5, 2.0]), HMC(0.1, 5), 1000)
```


```julia
using Turing

# Initialize a NamedTuple containing our data variables.
data = (x = [1.5, 2.0],)

# Create the model function.
mf(vi, sampler, ctx, model) = begin
    # Set the accumulated logp to zero.
    resetlogp!(vi)
    x = model.args.x

    # Assume s has an InverseGamma distribution.
    s, lp = Turing.Inference.tilde(
        ctx,
        sampler,
        InverseGamma(2, 3),
        Turing.@varname(s),
        (),
        vi,
    )

    # Add the lp to the accumulated logp.
    acclogp!(vi, lp)

    # Assume m has a Normal distribution.
    m, lp = Turing.Inference.tilde(
        ctx,
        sampler,
        Normal(0, sqrt(s)),
        Turing.@varname(m),
        (),
        vi,
    )

    # Add the lp to the accumulated logp.
    acclogp!(vi, lp)

    # Observe each value of x[i], according to a
    # Normal distribution.
    lp = Turing.Inference.dot_tilde(ctx, sampler, Normal(m, sqrt(s)), x, vi)
    acclogp!(vi, lp)
end

# Instantiate a Model object.
model = DynamicPPL.Model(mf, data, DynamicPPL.ModelGen{()}(nothing, nothing))

# Sample the model.
chain = sample(model, HMC(0.1, 5), 1000)
```


## Task Copying


Turing [copies](https://github.com/JuliaLang/julia/issues/4085) Julia tasks to deliver efficient inference algorithms, but it also provides alternative slower implementation as a fallback. Task copying is enabled by default. Task copying requires us to use the `CTask` facility which is provided by [Libtask](https://github.com/TuringLang/Libtask.jl) to create tasks.


