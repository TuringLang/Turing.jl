---
title: Library
permalink: /docs/library/
toc: true
---



<a id='Modelling-1'></a>

## Modelling

### <a id='Turing.Core.@model' href='#Turing.Core.@model'>#</a> **`Turing.Core.@model`** &mdash; *Macro*.


```julia
@model(body)
```

Macro to specify a probabilistic model.

Example:

Model definition:

```julia
@model model_generator(x = default_x, y) = begin
    ...
end
```

Expanded model definition

```julia
# Allows passing arguments as kwargs
model_generator(; x = nothing, y = nothing)) = model_generator(x, y)
function model_generator(x = nothing, y = nothing)
    pvars, dvars = Turing.get_vars(Tuple{:x, :y}, (x = x, y = y))
    data = Turing.get_data(dvars, (x = x, y = y))
    defaults = Turing.get_default_values(dvars, (x = default_x, y = nothing))

    inner_function(sampler::Turing.AbstractSampler, model) = inner_function(model)
    function inner_function(model)
        return inner_function(Turing.VarInfo(), Turing.SampleFromPrior(), model)
    end
    function inner_function(vi::Turing.VarInfo, model)
        return inner_function(vi, Turing.SampleFromPrior(), model)
    end
    # Define the main inner function
    function inner_function(vi::Turing.VarInfo, sampler::Turing.AbstractSampler, model)
        local x
        if isdefined(model.data, :x)
            x = model.data.x
        else
            x = model_defaults.x
        end
        local y
        if isdefined(model.data, :y)
            y = model.data.y
        else
            y = model.defaults.y
        end

        vi.logp = zero(Real)
        ...
    end
    model = Turing.Model{pvars, dvars}(inner_function, data, defaults)
    return model
end
```

Generating a model: `model_generator(x_value)::Model`.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/core/compiler.jl#L158-L214' class='documenter-source'>source</a><br>


<a id='Samplers-1'></a>

## Samplers

### <a id='Turing.Sampler' href='#Turing.Sampler'>#</a> **`Turing.Sampler`** &mdash; *Type*.


```julia
Sampler{T}
```

Generic interface for implementing inference algorithms. An implementation of an algorithm should include the following:

1. A type specifying the algorithm and its parameters, derived from InferenceAlgorithm
2. A method of `sample` function that produces results of inference, which is where actual inference happens.

Turing translates models to chunks that call the modelling functions at specified points. The dispatch is based on the value of a `sampler` variable. To include a new inference algorithm implements the requirements mentioned above in a separate file, then include that file at the end of this one.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/Turing.jl#L92-L105' class='documenter-source'>source</a><br>

### <a id='Turing.Inference.Gibbs' href='#Turing.Inference.Gibbs'>#</a> **`Turing.Inference.Gibbs`** &mdash; *Type*.


```julia
Gibbs(n_iters, algs...)
```

Compositional MCMC interface. Gibbs sampling combines one or more sampling algorithms, each of which samples from a different set of variables in a model.

Example:

```julia
@model gibbs_example(x) = begin
    v1 ~ Normal(0,1)
    v2 ~ Categorical(5)
        ...
end

# Use PG for a 'v2' variable, and use HMC for the 'v1' variable.
# Note that v2 is discrete, so the PG sampler is more appropriate
# than is HMC.
alg = Gibbs(1000, HMC(1, 0.2, 3, :v1), PG(20, 1, :v2))
```

Tips:

  * `HMC` and `NUTS` are fast samplers, and can throw off particle-based

methods like Particle Gibbs. You can increase the effectiveness of particle sampling by including more particles in the particle sampler.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/inference/gibbs.jl#L5-L30' class='documenter-source'>source</a><br>

### <a id='Turing.Inference.HMC' href='#Turing.Inference.HMC'>#</a> **`Turing.Inference.HMC`** &mdash; *Type*.


```julia
HMC(n_iters::Int, ϵ::Float64, n_leapfrog::Int)
```

Hamiltonian Monte Carlo sampler.

Arguments:

  * `n_iters::Int` : The number of samples to pull.
  * `ϵ::Float64` : The leapfrog step size to use.
  * `n_leapfrog::Int` : The number of leapfrop steps to use.

Usage:

```julia
HMC(1000, 0.05, 10)
```

Tips:

  * If you are receiving gradient errors when using `HMC`, try reducing the

`step_size` parameter.

```julia
# Original step_size
sample(gdemo([1.5, 2]), HMC(1000, 0.1, 10))

# Reduced step_size.
sample(gdemo([1.5, 2]), HMC(1000, 0.01, 10))
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/inference/hmc.jl#L5-L34' class='documenter-source'>source</a><br>

### <a id='Turing.Inference.HMCDA' href='#Turing.Inference.HMCDA'>#</a> **`Turing.Inference.HMCDA`** &mdash; *Type*.


```julia
HMCDA(n_iters::Int, n_adapts::Int, δ::Float64, λ::Float64; init_ϵ::Float64=0.1)
```

Hamiltonian Monte Carlo sampler with Dual Averaging algorithm.

Usage:

```julia
HMCDA(1000, 200, 0.65, 0.3)
```

Arguments:

  * `n_iters::Int` : Number of samples to pull.
  * `n_adapts::Int` : Numbers of samples to use for adaptation.
  * `δ::Float64` : Target acceptance rate. 65% is often recommended.
  * `λ::Float64` : Target leapfrop length.
  * `init_ϵ::Float64=0.1` : Inital step size; 0 means automatically search by Turing.

For more information, please view the following paper ([arXiv link](https://arxiv.org/abs/1111.4246)):

  * Hoffman, Matthew D., and Andrew Gelman. "The No-U-turn sampler: adaptively setting path lengths in Hamiltonian Monte Carlo." Journal of Machine Learning Research 15, no. 1 (2014): 1593-1623.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/inference/hmc.jl#L65-L89' class='documenter-source'>source</a><br>

### <a id='Turing.Inference.IPMCMC' href='#Turing.Inference.IPMCMC'>#</a> **`Turing.Inference.IPMCMC`** &mdash; *Type*.


```julia
IPMCMC(n_particles::Int, n_iters::Int, n_nodes::Int, n_csmc_nodes::Int)
```

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables must be stored in a [`TArray`](@ref) object.

Usage:

```julia
IPMCMC(100, 100, 4, 2)
```

Arguments:

  * `n_particles::Int` : Number of particles to use.
  * `n_iters::Int` : Number of iterations to employ.
  * `n_nodes::Int` : The number of nodes running SMC and CSMC.
  * `n_csmc_nodes::Int` : The number of CSMC nodes.

```

A paper on this can be found [here](https://arxiv.org/abs/1602.05128).


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/inference/AdvancedSMC.jl#L466-L489' class='documenter-source'>source</a><br>

### <a id='Turing.Inference.IS' href='#Turing.Inference.IS'>#</a> **`Turing.Inference.IS`** &mdash; *Type*.


```julia
IS(n_particles::Int)
```

Importance sampling algorithm.

Note that this method is particle-based, and arrays of variables must be stored in a [`TArray`](@ref) object.

Arguments:

  * `n_particles` is the number of particles to use.

Usage:

```julia
IS(1000)
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
    s ~ InverseGamma(2,3)
    m ~ Normal(0,sqrt.(s))
    x[1] ~ Normal(m, sqrt.(s))
    x[2] ~ Normal(m, sqrt.(s))
    return s, m
end

sample(gdemo([1.5, 2]), IS(1000))
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/inference/is.jl#L1-L33' class='documenter-source'>source</a><br>

### <a id='Turing.Inference.MH' href='#Turing.Inference.MH'>#</a> **`Turing.Inference.MH`** &mdash; *Type*.


```julia
MH(n_iters::Int)
```

Metropolis-Hastings sampler.

Usage:

```julia
MH(100, (:m, (x) -> Normal(x, 0.1)))
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

chn = sample(gdemo([1.5, 2]), MH(1000))
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/inference/mh.jl#L1-L26' class='documenter-source'>source</a><br>

### <a id='Turing.Inference.NUTS' href='#Turing.Inference.NUTS'>#</a> **`Turing.Inference.NUTS`** &mdash; *Type*.


```julia
NUTS(n_iters::Int, n_adapts::Int, δ::Float64)
```

No-U-Turn Sampler (NUTS) sampler.

Usage:

```julia
NUTS(1000, 200, 0.6j_max)
```

Arguments:

  * `n_iters::Int` : The number of samples to pull.
  * `n_adapts::Int` : The number of samples to use with adapatation.
  * `δ::Float64` : Target acceptance rate.
  * `max_depth::Float64` : Maximum doubling tree depth.
  * `Δ_max::Float64` : Maximum divergence during doubling tree.
  * `init_ϵ::Float64` : Inital step size; 0 means automatically search by Turing.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/inference/hmc.jl#L137-L157' class='documenter-source'>source</a><br>

### <a id='Turing.Inference.PG' href='#Turing.Inference.PG'>#</a> **`Turing.Inference.PG`** &mdash; *Type*.


```julia
PG(n_particles::Int, n_iters::Int)
```

Particle Gibbs sampler.

Note that this method is particle-based, and arrays of variables must be stored in a [`TArray`](@ref) object.

Usage:

```julia
PG(100, 100)
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/inference/AdvancedSMC.jl#L86-L99' class='documenter-source'>source</a><br>

### <a id='Turing.Inference.PMMH' href='#Turing.Inference.PMMH'>#</a> **`Turing.Inference.PMMH`** &mdash; *Type*.


```julia
PMMH(n_iters::Int, smc_alg:::SMC, parameters_algs::Tuple{MH})
```

Particle independant Metropolis–Hastings and Particle marginal Metropolis–Hastings samplers.

Note that this method is particle-based, and arrays of variables must be stored in a [`TArray`](@ref) object.

Usage:

```julia
alg = PMMH(100, SMC(20, :v1), MH(1,:v2))
alg = PMMH(100, SMC(20, :v1), MH(1,(:v2, (x) -> Normal(x, 1))))
```

Arguments:

  * `n_iters::Int` : Number of iterations to run.
  * `smc_alg:::SMC` : An [`SMC`]({{site.baseurl}}/docs/library/#Turing.Inference.SMC) algorithm to use.
  * `parameters_algs::Tuple{MH}` : An [`MH`]({{site.baseurl}}/docs/library/#Turing.Inference.MH) algorithm, which includes a

sample space specification.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/inference/AdvancedSMC.jl#L275-L297' class='documenter-source'>source</a><br>

### <a id='Turing.Inference.SGHMC' href='#Turing.Inference.SGHMC'>#</a> **`Turing.Inference.SGHMC`** &mdash; *Type*.


```julia
SGHMC(n_iters::Int, learning_rate::Float64, momentum_decay::Float64)
```

Stochastic Gradient Hamiltonian Monte Carlo sampler.

Usage:

```julia
SGHMC(1000, 0.01, 0.1)
```

Arguments:

  * `n_iters::Int` : Number of samples to pull.
  * `learning_rate::Float64` : The learning rate.
  * `momentum_decay::Float64` : Momentum decay variable.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/inference/sghmc.jl#L10-L27' class='documenter-source'>source</a><br>

### <a id='Turing.Inference.SGLD' href='#Turing.Inference.SGLD'>#</a> **`Turing.Inference.SGLD`** &mdash; *Type*.


```julia
SGLD(n_iters::Int, ϵ::Float64)
```

Stochastic Gradient Langevin Dynamics sampler.

Usage:

```julia
SGLD(1000, 0.5)
```

Arguments:

  * `n_iters::Int` : Number of samples to pull.
  * `ϵ::Float64` : The scaling factor for the learing rate.

Reference:

Welling, M., & Teh, Y. W. (2011).  Bayesian learning via stochastic gradient Langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11) (pp. 681-688).


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/inference/sghmc.jl#L115-L135' class='documenter-source'>source</a><br>

### <a id='Turing.Inference.SMC' href='#Turing.Inference.SMC'>#</a> **`Turing.Inference.SMC`** &mdash; *Type*.


```julia
SMC(n_particles::Int)
```

Sequential Monte Carlo sampler.

Note that this method is particle-based, and arrays of variables must be stored in a [`TArray`](@ref) object.

Usage:

```julia
SMC(1000)
```


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/inference/AdvancedSMC.jl#L9-L22' class='documenter-source'>source</a><br>


<a id='Data-Structures-1'></a>

## Data Structures


!!! warning "Missing docstring."
    Missing docstring for `TArray`. Check Documenter's build log for details.



<a id='Utilities-1'></a>

## Utilities


!!! warning "Missing docstring."
    Missing docstring for `tzeros`. Check Documenter's build log for details.



<a id='Index-1'></a>

## Index

- [`Turing.Inference.Gibbs`]({{site.baseurl}}/docs/library/#Turing.Inference.Gibbs)
- [`Turing.Inference.HMC`]({{site.baseurl}}/docs/library/#Turing.Inference.HMC)
- [`Turing.Inference.HMCDA`]({{site.baseurl}}/docs/library/#Turing.Inference.HMCDA)
- [`Turing.Inference.IPMCMC`]({{site.baseurl}}/docs/library/#Turing.Inference.IPMCMC)
- [`Turing.Inference.IS`]({{site.baseurl}}/docs/library/#Turing.Inference.IS)
- [`Turing.Inference.MH`]({{site.baseurl}}/docs/library/#Turing.Inference.MH)
- [`Turing.Inference.NUTS`]({{site.baseurl}}/docs/library/#Turing.Inference.NUTS)
- [`Turing.Inference.PG`]({{site.baseurl}}/docs/library/#Turing.Inference.PG)
- [`Turing.Inference.PMMH`]({{site.baseurl}}/docs/library/#Turing.Inference.PMMH)
- [`Turing.Inference.SGHMC`]({{site.baseurl}}/docs/library/#Turing.Inference.SGHMC)
- [`Turing.Inference.SGLD`]({{site.baseurl}}/docs/library/#Turing.Inference.SGLD)
- [`Turing.Inference.SMC`]({{site.baseurl}}/docs/library/#Turing.Inference.SMC)
- [`Turing.RandomMeasures.ChineseRestaurantProcess`]({{site.baseurl}}/docs/library/#Turing.RandomMeasures.ChineseRestaurantProcess)
- [`Turing.RandomMeasures.DirichletProcess`]({{site.baseurl}}/docs/library/#Turing.RandomMeasures.DirichletProcess)
- [`Turing.RandomMeasures.PitmanYorProcess`]({{site.baseurl}}/docs/library/#Turing.RandomMeasures.PitmanYorProcess)
- [`Turing.RandomMeasures.SizeBiasedSamplingProcess`]({{site.baseurl}}/docs/library/#Turing.RandomMeasures.SizeBiasedSamplingProcess)
- [`Turing.RandomMeasures.StickBreakingProcess`]({{site.baseurl}}/docs/library/#Turing.RandomMeasures.StickBreakingProcess)
- [`Turing.Sampler`]({{site.baseurl}}/docs/library/#Turing.Sampler)
- [`Turing.RandomMeasures._logpdf_table`]({{site.baseurl}}/docs/library/#Turing.RandomMeasures._logpdf_table-Union{Tuple{T}, Tuple{AbstractRandomProbabilityMeasure,T}} where T<:AbstractArray{Int64,1})
- [`Turing.Core.@model`]({{site.baseurl}}/docs/library/#Turing.Core.@model)


<a id='RandomMeasures-1'></a>

## RandomMeasures

### <a id='Turing.RandomMeasures.ChineseRestaurantProcess' href='#Turing.RandomMeasures.ChineseRestaurantProcess'>#</a> **`Turing.RandomMeasures.ChineseRestaurantProcess`** &mdash; *Type*.


```julia
ChineseRestaurantProcess(rpm, m)
```

The *Chinese Restaurant Process* for random probability measures `rpm` with counts `m`.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/stdlib/RandomMeasures.jl#L46-L50' class='documenter-source'>source</a><br>

### <a id='Turing.RandomMeasures.DirichletProcess' href='#Turing.RandomMeasures.DirichletProcess'>#</a> **`Turing.RandomMeasures.DirichletProcess`** &mdash; *Type*.


```julia
DirichletProcess(α)
```

The *Dirichlet Process* with concentration parameter `α`. Samples from the Dirichlet process can be constructed via the following representations.

*Size-Biased Sampling Process*

$$
j_k \sim Beta(1, \alpha) * surplus
$$

*Stick-Breaking Process*

$$
v_k \sim Beta(1, \alpha)
$$

*Chinese Restaurant Process*

$$
p(z_n = k | z_{1:n-1}) \propto \begin{cases} 
        \frac{m_k}{n-1+\alpha}, \text{if} m_k > 0\\ 
        \frac{\alpha}{n-1+\alpha}
    \end{cases}
$$

For more details see: https://www.stats.ox.ac.uk/~teh/research/npbayes/Teh2010a.pdf


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/stdlib/RandomMeasures.jl#L92-L117' class='documenter-source'>source</a><br>

### <a id='Turing.RandomMeasures.PitmanYorProcess' href='#Turing.RandomMeasures.PitmanYorProcess'>#</a> **`Turing.RandomMeasures.PitmanYorProcess`** &mdash; *Type*.


```julia
PitmanYorProcess(d, θ, t)
```

The *Pitman-Yor Process* with discount `d`, concentration `θ` and `t` already drawn atoms. Samples from the *Pitman-Yor Process* can be constructed via the following representations.

*Size-Biased Sampling Process*

$$
j_k \sim Beta(1-d, \theta + t*d) * surplus
$$

*Stick-Breaking Process*

$$
v_k \sim Beta(1-d, \theta + t*d)
$$

*Chinese Restaurant Process*

$$
p(z_n = k | z_{1:n-1}) \propto \begin{cases} 
        \frac{m_k - d}{n+\theta}, \text{if} m_k > 0\\ 
        \frac{\theta + d*t}{n+\theta}
    \end{cases}
$$

For more details see: https://en.wikipedia.org/wiki/Pitman–Yor_process


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/stdlib/RandomMeasures.jl#L154-L179' class='documenter-source'>source</a><br>

### <a id='Turing.RandomMeasures.SizeBiasedSamplingProcess' href='#Turing.RandomMeasures.SizeBiasedSamplingProcess'>#</a> **`Turing.RandomMeasures.SizeBiasedSamplingProcess`** &mdash; *Type*.


```julia
SizeBiasedSamplingProcess(rpm, surplus)
```

The *Size-Biased Sampling Process* for random probability measures `rpm` with a surplus mass of `surplus`.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/stdlib/RandomMeasures.jl#L17-L21' class='documenter-source'>source</a><br>

### <a id='Turing.RandomMeasures.StickBreakingProcess' href='#Turing.RandomMeasures.StickBreakingProcess'>#</a> **`Turing.RandomMeasures.StickBreakingProcess`** &mdash; *Type*.


```julia
StickBreakingProcess(rpm)
```

The *Stick-Breaking Process* for random probability measures `rpm`.


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/stdlib/RandomMeasures.jl#L32-L36' class='documenter-source'>source</a><br>

### <a id='Turing.RandomMeasures._logpdf_table-Union{Tuple{T}, Tuple{AbstractRandomProbabilityMeasure,T}} where T<:AbstractArray{Int64,1}' href='#Turing.RandomMeasures._logpdf_table-Union{Tuple{T}, Tuple{AbstractRandomProbabilityMeasure,T}} where T<:AbstractArray{Int64,1}'>#</a> **`Turing.RandomMeasures._logpdf_table`** &mdash; *Method*.


```julia
_logpdf_table(d<:AbstractRandomProbabilityMeasure, m<:AbstractVector{Int})
```

Parameters:

  * `d`: Random probability measure, e.g. DirichletProcess
  * `m`: Cluster counts


<a target='_blank' href='https://github.com/TuringLang/Turing.jl/blob/e29f651dc7e7d1f7d9d32cc35ece57ae6adb7474/src/stdlib/RandomMeasures.jl#L57-L65' class='documenter-source'>source</a><br>

