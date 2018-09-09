# Advanced Usage

## How to define a customized distribution

Turing.jl supports the use of distributions from the Distributions.jl package. By extension it also supports the use of customized distributions, by defining them as sub-types of `Distribution` type of the Distributions.jl package, as well as corresponding functions.

Below shows a workflow of how to define a customized distribution, using the flat prior as a simple example.

### 1. Define the distribution type

The first thing to do is to define a type of the distribution, as a subtype of a corresponding distribution type in the Distributions.jl package.

```julia
immutable Flat <: ContinuousUnivariateDistribution
end
```

### 2. Define functions for randomness

The second thing to do is to define `rand()` and `logpdf()`, which will be used to run the model.

```julia
Distributions.rand(d::Flat) = rand()
Distributions.logpdf{T<:Real}(d::Flat, x::T) = zero(x)
```

### 3. Define helper functions

In most cases, it may be required to define helper functions.

#### 3.1 Domain transformation

Some helper functions will be used domain transformation. For univariate distributions, the necessary ones are `minimum()` and `maximum()`.

```julia
Distributions.minimum(d::Flat) = -Inf
Distributions.maximum(d::Flat) = +Inf
```

Functions for domain transformation which may be required from multi-variate or matrix-variate distributions are `size(d)`, `link(d, x)` and `invlink(d, x)`. Please see `src/samplers/support/transform.jl` for examples.

#### 3.2 Vectorization support

Turing.jl supports a vectorization syntax `rv ~ [distribution]`, which requires `rand()` and `logpdf()` to be called on multiple data points. The functions for `Flat` are shown below.

```julia
Distributions.rand(d::Flat, n::Int) = Vector([rand() for _ = 1:n])
Distributions.logpdf{T<:Real}(d::Flat, x::Vector{T}) = zero(x)
```

## Avoid using `@model` macro

When integrating Turing.jl with other libraries, it's usually necessary to avoid using the `@model` macro. To achieve this, one needs to understand the `@model` macro, which basically works as a closure and generates an amended function by

1. Assigning the arguments to corresponding local variables;
2. Adding two keyword arguments `vi=VarInfo()` and `sampler=nothing` to the scope
3. Forcing the function to return `vi`

Thus by doing these three steps manually, one could get rid of the `@model` macro. Taking the `gdemo` model as an example, the two code sections below (macro and macro-free) have the same effect.

```julia
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end
mf = gdemo([1.5, 2.0])
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

## Task copying

Turing [copies](https://github.com/JuliaLang/julia/issues/4085) Julia tasks to deliver efficient inference algorithms, but it also provides alternative slower implementation as a fallback. Task copying is enabled by default. Task copying requires building a small C program, which should be done automatically on Linux and Mac systems that have GCC and Make installed.
