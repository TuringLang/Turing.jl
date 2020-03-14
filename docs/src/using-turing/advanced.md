---
title: Advanced Usage
---

# Advanced Usage

## How to Define a Customized Distribution


Turing.jl supports the use of distributions from the Distributions.jl package. By extension it also supports the use of customized distributions, by defining them as subtypes of `Distribution` type of the Distributions.jl package, as well as corresponding functions.


Below shows a workflow of how to define a customized distribution, using a flat prior as a simple example.


### 1. Define the Distribution Type


First, define a type of the distribution, as a subtype of a corresponding distribution type in the Distributions.jl package.


```julia
struct Flat <: ContinuousUnivariateDistribution
  …
end
```


### 2. Implement Sampling and Evaluation of the log-pdf


Second, define `rand` and `logpdf`, which will be used to run the model.


```julia
Distributions.rand(rng::AbstractRNG, d::Flat) = rand(rng)
Distributions.logpdf(d::Flat, x::Real) = zero(x)
```


### 3. Define Helper Functions


In most cases, it may be required to define helper functions, such as the `minimum`, `maximum`, `rand`, and `logpdf` functions, among others.


#### 3.1 Domain Transformation


Some helper functions are necessary for domain transformation. For univariate distributions, the necessary ones to implement are `minimum` and `maximum`.


```julia
Distributions.minimum(d::Flat) = -Inf
Distributions.maximum(d::Flat) = +Inf
```


Functions for domain transformation which may be required by multivariate or matrix-variate distributions are `size`, `link` and `invlink`. Please see Turing's [`transform.jl`](https://github.com/TuringLang/Turing.jl/blob/master/src/utilities/transform.jl) for examples.


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
    acclogp!(vi,
        Turing.Inference.dot_tilde(
            ctx, 
            sampler, 
            Normal(m, sqrt(s)), 
            x, 
            vi,
        ),
    )
end

# Instantiate a Model object.
model = DynamicPPL.Model(mf, data, DyanamicPPL.ModelGen{()}(nothing, nothing))

# Sample the model.
chain = sample(model, HMC(0.1, 5), 1000)
```


## Task Copying


Turing [copies](https://github.com/JuliaLang/julia/issues/4085) Julia tasks to deliver efficient inference algorithms, but it also provides alternative slower implementation as a fallback. Task copying is enabled by default. Task copying requires we use the `CTask` facility which is provided by [Libtask](https://github.com/TuringLang/Libtask.jl) to create tasks.


## Maximum a Posteriori Estimation


Turing does not currently have built-in methods for calculating the [maximum a posteriori](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) (MAP) for a model. This is a goal for Turing's implementation (see [this issue](https://github.com/TuringLang/Turing.jl/issues/605)), but for the moment, we present here a method for estimating the MAP using [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).


```julia
using Turing

# Define the simple gdemo model.
@model gdemo(x, y) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
    return s, m
end

function get_nlogp(model)
    # Construct a trace struct
    vi = Turing.VarInfo(model)

    # Define a function to optimize.
    function nlogp(sm)
        spl = Turing.SampleFromPrior()
        new_vi = Turing.VarInfo(vi, spl, sm)
        model(new_vi, spl)
        -getlogp(new_vi)
    end

    return nlogp
end

# Define our data points.
x = 1.5
y = 2.0
model = gdemo(x, y)
nlogp = get_nlogp(model)

# Import Optim.jl.
using Optim

# Create a starting point, call the optimizer.
sm_0 = [1.0, 1.0]
lb = [0.0, -Inf]
ub = [Inf, Inf]
result = optimize(nlogp, lb, ub, sm_0, Fminbox())
```

## Parallel Sampling


Turing does not natively support parallel sampling. Currently, users must perform additional structual support. Note that running chains in parallel may cause unintended issues.


Below is an example of how to run samplers in parallel. Note that each process must be given a separate seed, otherwise the samples generated by independent processes will be equivalent and unhelpful to inference.


```julia
using Distributed
addprocs(4)
@everywhere using Turing, Random

# Set the progress to false to avoid too many notifications.
@everywhere turnprogress(false)

# Set a different seed for each process.
for i in procs()
    @fetch @spawnat i Random.seed!(rand(Int64))
end

# Define the model using @everywhere.
@everywhere @model gdemo(x) = begin
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    for i in eachindex(x)
        x[i] ~ Normal(m, sqrt(s))
    end
end

# Sampling setup.
num_chains = 4
sampler = NUTS(0.65)
model = gdemo([1.2, 3.5]

# Run all samples.
chns = reduce(chainscat, pmap(x->sample(model, sampler, 1000), 1:num_chains))
```
