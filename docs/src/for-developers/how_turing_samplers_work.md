

| title                    | toc  |
| ------------------------ | ---- |
| How Turing samplers work | true |

# How Turing samplers work

Prerequisite: [Interface guide](https://turing.ml/dev/docs/for-developers/interface).

Consider the following code:

```julia
@model function gdemo(x, y)
    s ~ InverseGamma(2, 3)
    m ~ Normal(0, sqrt(s))
    x ~ Normal(m, sqrt(s))
    y ~ Normal(m, sqrt(s))
end

mod = gdemo(1.5, 2)
alg = IS()
n_samples = 1000

chn = sample(mod, alg, n_samples)
```

The function `sample` is part of the AbstractMCMC interface. As explained in the interface guide, building a a sampling method that can be used by `sample` consists in overloading the structs and functions in `AbstractMCMC`. The interface guide also gives a standalone example of their implementation, [`AdvancedMH.jl`](). 

Turing sampling methods (most of which are written [here](https://github.com/TuringLang/Turing.jl/tree/master/src/inference)) also implement `AbstractMCMC`. Turing defines a particular architecture for `AbstractMCMC` implementations, that enables working with models defined by the `@model` macro, and uses DynamicPPL as a backend. The goal of this page is to describe this architecture, and how you would go about implementing your own sampling method in Turing. I don't go into all the details: for instance, I don't address selectors or parallelism.

## 1. Define a `Sampler`

Recall the last line of the above code block:

```julia
chn = sample(mod, alg, n_samples)
```

Here `sample` takes as arguments a **model** `mod`, an **algorithm** `alg`, and a **number of samples** `n_samples`, and returns an instance `chn` of `Chains` which can be analysed using the functions in `MCMCChains`.

### Models

To define a **model**, you declare a joint distribution on variables in the `@model` macro, and specify which variables are observed and which should be inferred, as well as the value of the observed variables. Thus, when implementing Importance Sampling,

```julia
mod = gdemo(1.5, 2)
```

creates an instance `mod` of the struct `Model`, which corresponds to the observations of a value of `1.5` for `x`, and a value of `2` for `y`.

This is all handled by DynamicPPL, more specifically [here](https://github.com/TuringLang/DynamicPPL.jl/blob/master/src/model.jl). I will return to how models are used to inform sampling algorithms [below](). TODO: link

### Algorithms

An **algorithm** is just a sampling method: in Turing, it is a subtype of the abstract type `InferenceAlgorithm`. Defining an algorithm may require specifying a few high-level parameters. For example, "Hamiltonian Monte-Carlo" may be too vague, but "Hamiltonian Monte Carlo with  10 leapfrog steps per proposal and a stepsize of 0.01" is an algorithm. "Metropolis-Hastings" may be too vague, but "Metropolis-Hastings with proposal distribution `p`" is an algorithm. $\epsilon$

Thus

```julia
stepsize = 0.01
L = 10
alg = HMC(stepsize, L)
```

defines a Hamiltonian Monte-Carlo algorithm, an instance of `HMC`, which is a subtype of `InferenceAlgorithm`.

In the case of Importance Sampling, there is no need to specify additional parameters:

```julia
alg = IS()
```

defines an Importance Sampling algorithm, an instance of `IS` which is a subtype of `InferenceAlgorithm`. 

When creating your own Turing sampling method, you must therefore build a subtype of `InferenceAlgorithm` corresponding to your method.

### Samplers

Samplers are **not** the same as algorithms. An algorithm is a generic sampling method, a sampler is an object that stores information about how algorithm and model interact during sampling, and is modified as sampling progresses. The `Sampler` struct is defined in DynamicPPL.

Turing implements `AbstractMCMC`'s `AbstractSampler` with the `Sampler` struct defined in `DynamicPPL`. The most important attributes of an instance `spl` of `Sampler` are:

* `spl.alg`: the sampling method used, an instance of a subtype of `InferenceAlgorithm`
* `spl.state`: information about the sampling process, see [below]() TODO: link

When you call `sample(mod, alg, n_samples)`, Turing first uses `model` and `alg` to build an instance `spl` of `Sampler` , then calls the native `AbstractMCMC` function `sample(mod, spl, n_samples)`. 

When you define your own Turing sampling method, you must therefore build: 

* a **sampler constructor** that uses a model and an algorithm to initialize an instance of `Sampler`. For Importance Sampling:

```julia
function Sampler(alg::IS, model::Model, s::Selector)
    info = Dict{Symbol, Any}()
    state = ISState(model)
    return Sampler(alg, info, s, state)
end
```

* a **state** struct implementing `AbstractSamplerState` corresponding to your method: we cover this in the following paragraph.

### States

```julia
mutable struct ISState{V<:VarInfo, F<:AbstractFloat} <: AbstractSamplerState
    vi                 ::  V
    final_logevidence  ::  F
end

# additional constructor
ISState(model::Model) = ISState(VarInfo(model), 0.0)
```

VarInfo contains all the important information about sampling: names of model parameters, the distributions from which they are sampled, the value of the samples, and other metadata.

As we will see below, many important steps during sampling correspond to queries or updates to `spl.state.vi`.

By default, you can use `SamplerState`, a concrete type extending `AbstractSamplerState` which has no field apart from `vi`.

![Untitled Diagram(1)](/Users/js/Downloads/Untitled Diagram(1).png)

## 2. Overload the functions used inside `mcmcsample`

A lot of the things here are method-specific. However Turing also has some functions that make it easier for you to implement these functions, for examples .

### Transitions

`AbstractMCMC` stores information corresponding to each individual sample in objects called `transition`, but does not specify what the structure of these objects could be. You could decide to implement a type `MyTransition` for transitions corresponding to the specifics of your methods. However, there are many situations in which the only information you need for each sample is:

* its value: $\theta$ 
* log of the joint probability of the observed data and this sample: `lp`

`Inference.jl` defines a struct `Transition`, which corresponds to this default situation, as well as a constructor that builds instances of `Transition` from instances of `Sampler`, by finding $\theta$ and `lp` in `spl.state.vi`.

Construct transition from a `spl`: just dump the contents of vi-made-nametuple into a transition 

### How `sample` works

A crude summary, which ignores things like parallelism, is the following. `sample` calls `mcmcsample`, which calls 

* `sample_init!` to set things up
* `step!` repeatedly to produce multiple new transitions
* `sample_end!` to perform operations once all samples have been obtained 
* `bundle_samples` to convert a vector of transitions into a more palatable type, for instance a `Chain`.

you can of course implement all of these functions, but `AbstractMCMC` as well 

## 3. Overload `assume` and `observe`

The functions mentioned above, such as `sample_init!`, `step!`, etc.,  must of course use information about the model in order to generate samples! In particular, these functions may need **samples from distributions** defined in the model, or to **evaluate the density of these distributions** at some values of the corresponding parameters or observations.

For an example of the former, consider **Importance Sampling** as defined in `is.jl`. This implementation of Importance Sampling uses the model prior distribution as a proposal distribution, and therefore requires **samples from the prior distribution** of the model. Another example is **Approximate Bayesian Computation**, which requires multiple **samples from the model prior and likelihood distributions** in order to generate a single sample.

An example of the latter is the **Metropolis-Hastings** algorithm. At every step of sampling from a target posterior $p(\theta \mid x_{\text{obs}})$, in order to compute the acceptance ratio, you need to **evaluate the model joint density** $p(\theta_{\text{prop}}, x_{\text{obs}})$ with $\theta_{\text{prop}}$ a sample from the proposal and $x_{\text{obs}}$ the observed data.

Below is still very preliminary:

This begs the question: how can these functions access model information during sampling? Recall that the model is stored as an instance `m` of `Model`. When the sampler needs to access model information, it executes the `m.f` function.

@model macro to instance of Model. main attribute of Model is the mf function. a large part of building the Model object is building this function from the lines in the @macro statement. 

among the arguments of `f` there is a sampler. this is because executing `f` runs the steps of the model in order and modifies the sampler for every tilde statement. how it modifies the sampler is defined by the assume and observe functions.

general idea: given a `Model` instance `m` and a sampler `spl`

* `step!` will call `m(..., spl, ...)` which is basically `m.f(..., spl, ...)`
* for every tilde statement in, `m.f(..., spl, ...)` will add model-related information (samples, model density evaluation, etc.) to `spl` (often to `spl.state.vi`). How does it do that?
  * recall that the code for `m.f(..., spl, ...)` is automatically generated by compilation of the `@model` macro
  * for every tilde statement in the `@model` declaration, this code contains a call to `assume(..., spl, ...)` if the variable on the LHS of the tilde is a **model parameter to infer**, and `observe(..., spl, ...)` if the variable on the LHS of the tilde is an **observation**
  * in the file corresponding to your sampling method (ie in `Turing.jl/src/inference/<your_method>.jl`), you have **overloaded** `assume` and `observe` (eg you have written code for `DynamicPPL.assume(rng, spl::Sampler{<:YourSamplingAlgorithm}, dist::Distribution, vn::VarName, vi)`, cf in `is.jl`) so that when `spl` uses your sampling method, `assume` and `observe` modify `spl` to include the information and samples that you care about!
