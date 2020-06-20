# How Turing samplers work

Sources: https://github.com/TuringLang/Turing.jl/issues/895

Prerequisite: [Interface guide](https://turing.ml/dev/docs/for-developers/interface).

The `AbstractMCMC` interface defines a function `sample` that takes as inputs a model, a sampler and a number of samples, and outputs a `Chains` object. The interface guide describes the roles of main structs and functions in `AbstractMCMC` and gives an example of their implementation, [`AdvancedMH.jl`](). 

Turing sampling methods (see code [here](https://github.com/TuringLang/Turing.jl/tree/master/src/inference)), also implement `AbstractMCMC`. Turing defines a particular architecture for `AbstractMCMC` implementations, that enables working with models defined by the `@model` macro, and uses DynamicPPL as a backend. The goal of this page is to describe this architecture, and how you would go about implementing your own sampling method in Turing. It doesn't go into all the details

## 1. Define a `Sampler`

Hello world:

Here `sample` takes as arguments a model, an algorithm, and a number of samples.

### Models

To define a model, you declare a joint distribution on variables in the `@model` macro, and specify which variables are observed and which should be inferred, as well as the value of the observed variables. Thus `<code>` creates an instance of the struct `Model`. This is all handled by DynamicPPL, more specifically [here](https://github.com/TuringLang/DynamicPPL.jl/blob/master/src/model.jl).

More about models below.

### Algorithms

An algorithm is just a sampling method. Defining an algorithm may require specifying a few high-level parameters. For example, "Hamiltonian Monte-Carlo" may be too vague, but "Hamiltonian Monte Carlo with  10 leapfrog steps per proposal and a stepsize of 0.01" is an algorithm. "Metropolis-Hastings" may be too vague, but "Metropolis-Hastings with proposal distribution `p`" is an algorithm. Thus `HMC()` creates an instance of `HMC`, which is a subtype of `InferenceAlgorithm`. 

When creating your own Turing sampling method, you must therefore build a subtype of `InferenceAlgorithm` corresponding to your method.

> Importance sampling example here: 

### Samplers

AbstractMCMC itself doesn't know about `InferenceAlgorithm`. Its native `sample` functions take as inputs (among other things) models (from `AbstractModel`) and **samplers** (from `AbstractSampler)`, not models and **algorithms**. Samplers are **not** the same as algorithms. An algorithm is a generic sampling method, a sampler is an object that stores information about how algorithm and model interact during sampling, and is modified as sampling progresses.

Turing implements `AbstractMCMC`'s `AbstractSampler` with the `Sampler` struct defined in `DynamicPPL`. The most important attributes of an instance `spl` of `Sampler` are:

* `spl.alg`: the sampling method used, an instance of a subtype of `InferenceAlgorithm`
* `spl.state`: information about the sampling process, eg which variables have been sampled so far, what has the

When you call `sample(model, alg, N)`, Turing first uses `model` and `alg` to build an instance `spl` of `Sampler` , then calls the native `AbstractMCMC` function `sample(model, spl, N)`. 

When you define your own Turing sampling method, you must therefore build: 

* a **state struct** implementing `AbstractSamplerState` corresponding to your method
* a **sampler constructor** that initializes an instance of `Sampler` from a model and an algorithm.

> Show sampler constructor for Importance sampling, mention 

### States

> Show `ISState` definition

VarInfo contains all the important information about sampling: names of model parameters, the distributions from which they are sampled, the value of the samples, and other metadata.

As we will see below, many important steps during sampling correspond to queries or updates to `spl.state.vi`.

By default, you can use `SamplerState` which *only* has a VarInfo attribute.



## 2. Implement the functions used inside `mcmcsample`

A lot of the things here are method-specific. However Turing also has some functions that make it easier for you to implement these functions, for examples .

### Transitions

`AbstractMCMC` stores information corresponding to each individual sample in objects called `transition`, but does not specify what the structure of these objects could be. You could decide to implement a type `MyTransition` for transitions corresponding to the specifics of your methods. However, there are many situations in which the only information you need for each sample is:

* its value: $\theta$ 
* log of the joint probability of the observed data and this sample: `lp`

`inference/Inference.jl` defines a struct `Transition`, which corresponds to this default situation, as well as a constructor that builds instances of `Transition` from instances of `Sampler`, by finding $\theta$ and `lp` in `spl.state.vi`.

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

For an example of the former, consider **Importance Sampling** as defined in `is.jl`. This implementation of importance sampling uses the model prior distribution as a proposal distribution, and therefore requires **samples from the prior distribution** of the model. Another example is **Approximate Bayesian Computation**, which requires multiple **samples from the model prior and likelihood distributions** in order to generate a single sample.

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

