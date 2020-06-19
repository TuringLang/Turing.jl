# How Turing samplers work

Sources: https://github.com/TuringLang/Turing.jl/issues/895

Prerequisite: [Interface guide](https://turing.ml/dev/docs/for-developers/interface).

The `AbstractMCMC` interface defines a function `sample` that takes as inputs a model, a sampler and a number of samples, and outputs a `Chains` object. The interface guide describes the roles of main structs and functions in `AbstractMCMC` and gives an example of their implementation, [`AdvancedMH.jl`](). 

Turing sampling methods (see code [here](https://github.com/TuringLang/Turing.jl/tree/master/src/inference)), also implement `AbstractMCMC`. Turing defines a particular architecture for `AbstractMCMC` implementations, that enables working with models defined by the `@model` macro, and uses DynamicPPL as a backend. The goal of this page is to describe this architecture, and how you would go about implementing your own sampling method in Turing. It doesn't go into all the details

## 1. Define a `Sampler`

Hello world:

Here `sample` takes as arguments a model, an algorithm, and a number of samples.

### Models

To define a model, you declare a joint distribution on variables in the `@model` macro, and specify which variables are observed and which should be inferred, as well as the value of the observed variables. Thus `<code>` creates an instance of `Model`. This is all handled by DynamicPPL.

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

### Transitions

abstractMCMC stores information corresponding to each individual sample in instances of Transition. 

construct transition from a spl: just dump the contents of vi-made-nametuple into a transition 

### How `sample` works

it calls `mcmcsample`which calls `sample_init!` to set things up, `step!` to produce new transitions, `sample_end!` to perform final operations, and `bundle_samples` to convert a vector of transitions into a more palatable type, for instance a `Chain`.

you get to implement all of these funtions. a lot of those will be method-specific, but there are things

## 3. Overload `assume` and `observe`

we often want specific information from the model. in general:

* sample from prior etc.
* compute log-likelihood

etc.
