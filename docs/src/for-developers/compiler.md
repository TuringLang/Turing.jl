---
title: Turing Compiler Design
---

In this section, the current design of Turing's model "compiler" is described which enables Turing to perform various types of Bayesian inference without changing the model definition. The "compiler" is essentially just a macro that rewrites the user's model definition to a function that generates a `Model` struct that Julia's dispatch can operate on and that Julia's compiler can successfully do type inference on for efficient machine code generation.

# Overview

The following terminology will be used in this section:

- `D`: observed data variables conditioned upon in the posterior,
- `P`: parameter variables distributed according to the prior distributions, these will also be referred to as random variables,
- `Model`: a fully defined probabilistic model with input data

`Turing`'s `@model` macro rewrites the user-provided function definition such that it can be used to instantiate a `Model` by passing in the observed data `D`.

The following are the main jobs of the `@model` macro:
1. Parse `~` and `.~` lines, e.g. `y .~ Normal.(c*x, 1.0)`
2. Figure out if a variable belongs to the data `D` and or to the parameters `P`
3. Enable the handling of missing data variables in `D` when defining a `Model` and treating them as parameter variables in `P` instead
4. Enable the tracking of random variables using the data structures `VarName` and `VarInfo`
5. Change `~`/`.~` lines with a variable in `P` on the LHS to a call to `tilde_assume` or `dot_tilde_assume`
6. Change `~`/`.~` lines with a variable in `D` on the LHS to a call to `tilde_observe` or `dot_tilde_observe`
7. Enable type stable automatic differentiation of the model using type parameters


## The model

A `model::Model` is a callable struct that one can sample from by calling
```julia
(model::Model)([rng, varinfo, sampler, context])
```
where `rng` is a random number generator (default: `Random.GLOBAL_RNG`), `varinfo` is a data structure that stores information
about the random variables (default: `DynamicPPL.VarInfo()`), `sampler` is a sampling algorithm (default: `DynamicPPL.SampleFromPrior()`),
and `context` is a sampling context that can, e.g., modify how the log probability is accumulated (default: `DynamicPPL.DefaultContext()`).

Sampling resets the log joint probability of `varinfo` and increases the evaluation counter of `sampler`. If `context` is a `LikelihoodContext`,
only the log likelihood will be accumulated. With the `DefaultContext` the log joint probability of `P` and `D` is accumulated.

The `Model` struct contains the three internal fields `f`, `args` and `defaults`.
When `model::Model` is called, then the internal function `model.f` is called as `model.f(rng, varinfo, sampler, context, model.args...)`
(for multithreaded sampling, instead of `varinfo` a threadsafe wrapper is passed to `model.f`).
The positional and keyword arguments that were passed to the user-defined model function when the model was created are saved as a `NamedTuple`
in `model.args`. The default values of the positional and keyword arguments of the user-defined model functions, if any, are saved as a `NamedTuple`
in `model.defaults`. They are used for constructing model instances with different arguments by the `logprob` and `prob` string macros.

# Example

Let's take the following model as an example:

```julia
@model function gauss(x = missing, y = 1.0, ::Type{TV} = Vector{Float64}) where {TV<:AbstractVector}
    if x === missing
        x = TV(undef, 3)
    end
    p = TV(undef, 2)
    p[1] ~ InverseGamma(2, 3)
    p[2] ~ Normal(0, 1.0)
    @. x[1:2] ~ Normal(p[2], sqrt(p[1]))
    x[3] ~ Normal()
    y ~ Normal(p[2], sqrt(p[1]))
end
```

The above call of the `@model` macro defines the function `gauss` with positional arguments `x`, `y`, and `::Type{TV}`, rewritten in
such a way that every call of it returns a `model::Model`. Note that only the function body is modified by the `@model` macro, and the
function signature is left untouched. It is also possible to implement models with keyword arguments such as

```julia
@model function gauss(::Type{TV} = Vector{Float64}; x = missing, y = 1.0) where {TV<:AbstractVector}
    ...
end
```

This would allow us to generate a model by calling `gauss(; x = rand(3))`.

If an argument has a default value `missing`, it is treated as a random variable. For variables which require an intialization because we
need to loop or broadcast over its elements, such as `x` above, the following needs to be done:
```julia
if x === missing
    x = ...
end
```
Note that since `gauss` behaves like a regular function it is possible to define additional dispatches in a second step as well. For
instance, we could achieve the same behaviour by

```julia
@model function gauss(x, y = 1.0, ::Type{TV} = Vector{Float64}) where {TV<:AbstractVector}
    p = TV(undef, 2)
    ...
end

function gauss(::Missing, y = 1.0, ::Type{TV} = Vector{Float64}) where {TV<:AbstractVector}
    return gauss(TV(undef, 3), y, TV)
end
```

If `x` is sampled as a whole from a distribution and not indexed, e.g., `x ~ Normal(...)` or `x ~ MvNormal(...)`,
there is no need to initialize it in an `if`-block.

## Step 1: Break up the model definition

First, the `@model` macro breaks up the user-provided function definition using `DynamicPPL.build_model_info`. This function
returns a dictionary consisting of:
- `allargs_exprs`: The expressions of the positional and keyword arguments, without default values.
- `allargs_syms`: The names of the positional and keyword arguments, e.g., `[:x, :y, :TV]` above.
- `allargs_namedtuple`: An expression that constructs a `NamedTuple` of the positional and keyword arguments, e.g., `:((x = x, y = y, TV = TV))` above.
- `defaults_namedtuple`: An expression that constructs a `NamedTuple` of the default positional and keyword arguments, if any, e.g., `:((x = missing, y = 1, TV = Vector{Float64}))` above.
- `modeldef`: A dictionary with the name, arguments, and function body of the model definition, as returned by `MacroTools.splitdef`.

## Step 2: Generate the body of the internal model function

In a second step, `DynamicPPL.generate_mainbody` generates the main part of the transformed function body using the user-provided function body
and the provided function arguments, without default values, for figuring out if a variable denotes an observation or a random variable.
Hereby the function `DynamicPPL.generate_tilde` replaces the `L ~ R` lines in the model and the function `DynamicPPL.generate_dot_tilde` replaces
the `@. L ~ R` and `L .~ R` lines in the model.

In the above example, `p[1] ~ InverseGamma(2, 3)` is replaced with something similar to
```julia
#= REPL[25]:6 =#
begin
    var"##tmpright#323" = InverseGamma(2, 3)
    var"##tmpright#323" isa Union{Distribution, AbstractVector{<:Distribution}} || throw(ArgumentError("Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions."))
    var"##vn#325" = (DynamicPPL.VarName)(:p, ((1,),))
    var"##inds#326" = ((1,),)
    p[1] = (DynamicPPL.tilde_assume)(_rng, _context, _sampler, var"##tmpright#323", var"##vn#325", var"##inds#326", _varinfo)
end
```
Here the first line is a so-called line number node that enables more helpful error messages by providing users with the exact location
of the error in their model definition. Then the right hand side (RHS) of the `~` is assigned to a variable (with an automatically generated name).
We check that the RHS is a distribution or an array of distributions, otherwise an error is thrown.
Next we extract a compact representation of the variable with its name and index (or indices). Finally, the `~` expression is replaced with
a call to `DynamicPPL.tilde_assume` since the compiler figured out that `p[1]` is a random variable using the following
heuristic:
1. If the symbol on the LHS of `~`, `:p` in this case, is not among the arguments to the model, `(:x, :y, :T)` in this case, it is a random variable. 
2. If the symbol on the LHS of `~`, `:p` in this case, is among the arguments to the model but has a value of `missing`, it is a random variable.
2. If the value of the LHS of `~`, `p[1]` in this case, is `missing`, then it is a random variable.
4. Otherwise, it is treated as an observation.

The `DynamicPPL.tilde_assume` function takes care of sampling the random variable, if needed, and updating its value and the accumulated log joint
probability in the `_varinfo` object. If `L ~ R` is an observation, `DynamicPPL.tilde_observe` is called with the same arguments except the
random number generator `_rng` (since observations are never sampled).

A similar transformation is performed for expressions of the form `@. L ~ R` and `L .~ R`. For instance,
`@. x[1:2] ~ Normal(p[2], sqrt(p[1]))` is replaced with
```julia
#= REPL[25]:8 =#
begin
    var"##tmpright#331" = Normal.(p[2], sqrt.(p[1]))
    var"##tmpright#331" isa Union{Distribution, AbstractVector{<:Distribution}} || throw(ArgumentError("Right-hand side of a ~ must be subtype of Distribution or a vector of Distributions."))
    var"##vn#333" = (DynamicPPL.VarName)(:x, ((1:2,),))
    var"##inds#334" = ((1:2,),)
    var"##isassumption#335" = begin
        let var"##vn#336" = (DynamicPPL.VarName)(:x, ((1:2,),))
            if !((DynamicPPL.inargnames)(var"##vn#336", _model)) || (DynamicPPL.inmissings)(var"##vn#336", _model)
                true
            else
                x[1:2] === missing
            end
        end
    end
    if var"##isassumption#335"
        x[1:2] .= (DynamicPPL.dot_tilde_assume)(_rng, _context, _sampler, var"##tmpright#331", x[1:2], var"##vn#333", var"##inds#334", _varinfo)
    else
        (DynamicPPL.dot_tilde_observe)(_context, _sampler, var"##tmpright#331", x[1:2], var"##vn#333", var"##inds#334", _varinfo)
    end
end
```
The main difference in the expanded code between `L ~ R` and `@. L ~ R` is that the former doesn't assume `L` to be defined, it can be a new Julia variable in the scope, while the latter assumes `L` already exists. Moreover, `DynamicPPL.dot_tilde_assume` and `DynamicPPL.dot_tilde_observe` are called
instead of `DynamicPPL.tilde_assume` and `DynamicPPL.tilde_observe`.

## Step 3: Replace the user-provided function body

Finally, we replace the user-provided function body using `DynamicPPL.build_output`. This function uses `MacroTools.combinedef` to reassemble
the user-provided function with a new function body. In the modified function body an anonymous function is created whose function body
was generated in step 2 above and whose arguments are
- a random number generator `_rng`,
- a model `_model`,
- a datastructure `_varinfo`,
- a sampler `_sampler`,
- a sampling context `_context`,
- and all positional and keyword arguments of the user-provided model function as positional arguments
without any default values. Finally, in the new function body a `model::Model` with this anonymous function as internal function is returned.

# `VarName`

In order to track random variables in the sampling process, `Turing` uses the struct `VarName{sym}` which acts as a random variable identifier generated at runtime. The `VarName` of a random variable is generated from the expression on the LHS of a `~` statement when the symbol on the LHS is in `P`. Every `vn::VarName{sym}` has a symbol `sym` which is the symbol of the Julia variable in the model that the random variable belongs to. For example, `x[1] ~ Normal()` will generate an instance of `VarName{:x}` assuming `x` is in `P`. Every `vn::VarName` also has a field `indexing` which stores the indices requires to access the random variable from the Julia variable indicated by `sym`. For example, `x[1] ~ Normal()` will generate a `vn::VarName{:x}` with `vn.indexing == "[1]"`. `VarName` also supports hierarchical arrays and range indexing. Some more examples:
- `x[1] ~ Normal()` will generate a `VarName{:x}` with `indexing == "[1]"`.
- `x[:,1] ~ MvNormal(zeros(2))` will generate a `VarName{:x}` with `indexing == "[Colon(),1]"`.
- `x[:,1][2] ~ Normal()` will generate a `VarName{:x}` with `indexing == "[Colon(),1][2]"`.

# `VarInfo`

## Overview

`VarInfo` is the data structure in `Turing` that facilitates tracking random variables and certain metadata about them that are required for sampling. For instance, the distribution of every random variable is stored in `VarInfo` because we need to know the support of every random variable when sampling using HMC for example. Random variables whose distributions have a constrained support are transformed using a bijector from [Bijectors.jl](https://github.com/TuringLang/Bijectors.jl) so that the sampling happens in the unconstrained space. Different samplers require different metadata about the random variables.

The definition of `VarInfo` in `Turing` is:
```
struct VarInfo{Tmeta, Tlogp} <: AbstractVarInfo
    metadata::Tmeta
    logp::Base.RefValue{Tlogp}
    num_produce::Base.RefValue{Int}
end
```
Based on the type of `metadata`, the `VarInfo` is either aliased `UntypedVarInfo` or `TypedVarInfo`. `metadata` can be either a subtype of the union type `Metadata` or a `NamedTuple` of multiple such subtypes. Let `vi` be an instance of `VarInfo`. If `vi isa VarInfo{<:Metadata}`, then it is called an `UntypedVarInfo`. If `vi isa VarInfo{<:NamedTuple}`, then `vi.metadata` would be a `NamedTuple` mapping each symbol in `P` to an instance of `Metadata`. `vi` would then be called a `TypedVarInfo`. The other fields of `VarInfo` include `logp` which is used to accumulate the log probability or log probability density of the variables in `P` and `D`. `num_produce` keeps track of how many observations have been made in the model so far. This is incremented when running a `~` statement when the symbol on the LHS is in `D`.

## `Metadata`

The `Metadata` struct stores some metadata about the random variables sampled. This helps
query certain information about a variable such as: its distribution, which samplers
sample this variable, its value and whether this value is transformed to real space or
not. Let `md` be an instance of `Metadata`:
- `md.vns` is the vector of all `VarName` instances. Let `vn` be an arbitrary element of `md.vns`
- `md.idcs` is the dictionary that maps each `VarName` instance to its index in
 `md.vns`, `md.ranges`, `md.dists`, `md.orders` and `md.flags`.
- `md.vns[md.idcs[vn]] == vn`.
- `md.dists[md.idcs[vn]]` is the distribution of `vn`.
- `md.gids[md.idcs[vn]]` is the set of algorithms used to sample `vn`. This is used in
 the Gibbs sampling process.
- `md.orders[md.idcs[vn]]` is the number of `observe` statements before `vn` is sampled.
- `md.ranges[md.idcs[vn]]` is the index range of `vn` in `md.vals`.
- `md.vals[md.ranges[md.idcs[vn]]]` is the linearized vector of values of corresponding to `vn`.
- `md.flags` is a dictionary of true/false flags. `md.flags[flag][md.idcs[vn]]` is the
 value of `flag` corresponding to `vn`.

Note that in order to make `md::Metadata` type stable, all the `md.vns` must have the same symbol and distribution type. However, one can have a single Julia variable, e.g. `x`, that is a matrix or a hierarchical array sampled in partitions, e.g. `x[1][:] ~ MvNormal(zeros(2), 1.0); x[2][:] ~ MvNormal(ones(2), 1.0)`. The symbol `x` can still be managed by a single `md::Metadata` without hurting the type stability since all the distributions on the RHS of `~` are of the same type. 

However, in `Turing` models one cannot have this restriction, so we must use a type unstable `Metadata` if we want to use one `Metadata` instance for the whole model. This is what `UntypedVarInfo` does. A type unstable `Metadata` will still work but will have inferior performance.

To strike a balance between flexibility and performance when constructing the `spl::Sampler` instance, the model is first run by sampling the parameters in `P` from their priors using an `UntypedVarInfo`, i.e. a type unstable `Metadata` is used for all the variables. Then once all the symbols and distribution types have been identified, a `vi::TypedVarInfo` is constructed where `vi.metadata` is a `NamedTuple` mapping each symbol in `P` to a specialized instance of `Metadata`. So as long as each symbol in `P` is sampled from only one type of distributions, `vi::TypedVarInfo` will have fully concretely typed fields which brings out the peak performance of Julia.
