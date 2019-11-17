---
title: Turing Compiler Design
---

In this section, I will describe the current design of Turing's model compiler which enables Turing to perform various types of Bayesian inference without changing the model definition. What we call "compiler" is essentially just a macro that transforms the user's code to something that Julia's dispatch can operate on and that Julia's compiler can successfully do type inference on for efficient machine code generation.

# Overview

The following terminology will be used in this section:

- `D`: observed data variables conditioned upon in the posterior,
- `P`: parameter variables distributed according to the posterior distribution, these will also be referred to as random variables,
- `Model`: a fully defined probabilistic model with input data, and
- `Model` generator: a function that can be used to instantiate a `Model` instance by inputing data `D`.

`Turing`'s `@model` macro defines a `Model` generator that can be used to instantiate a `Model` by passing in the observed data `D`.

# `@model` macro

The following are the main jobs of the `@model` macro:
1. Parse `~` lines, e.g. `y ~ Normal(c*x, 1.0)`
2. Figure out the set of symbols comprising the data `D` and the set of symbols comprising the parameters `P`
3. Enable the handling of missing data variables in `D` when defining a `Model` and treating them as parameter variables in `P` instead
4. Enable the tracking of random variables using the data structures `VarName` and `VarInfo`
5. Change `~` lines with a variable in `P` on the LHS to a call to the `assume` function
6. Change `~` lines with a variable in `D` on the LHS to a call to the `observe` function
7. Enable type stable automatic differentiation of the model using type parameters

Let's take the following model as an example:
```julia
@model gauss(x = zeros(2), y, ::Type{TV} = Vector{Float64}) where {TV <: AbstractVector} = begin
    p = TV(undef, 2)
    p[1] ~ InverseGamma(2, 3)
    p[2] ~ Normal(0, 1.0)
    for i in 1:length(x)
        x[i] ~ Normal(p[2], sqrt(p[1]))
    end
    y ~ Normal(p[2], sqrt(p[1]))
end
```
A `model::Model` can be defined using `gauss(rand(3), 1.0)` or `gauss(x = rand(3), y = 1.0)`. While constructing the model, some arguments can be ignored and will be treated as parameters in `P`, e.g. `gauss(rand(3))` will treat `y` as part of `P` and `gauss(y = 1.0)` will treat `x` as part of `P`. Note that if `x` is missing, inside the model we don't have access to `length(x)` and we don't know what `x` is anymore. This is the purpose of the "default value" of `x` in the model definition, `zeros(2)`, which tells us that `x` has length 2. To be more precise, `zeros(2)` will be called the default initialization of `x`.

The `@model` macro is defined as:
```julia
macro model(input_expr)
    build_model_info(input_expr) |> translate_tilde! |> update_args! |> build_output
end
```

## `build_model_info`

The first stop that the model definition takes is `build_model_info`. This function extracts some information from the model definition such as:
- `main_body`: the model body excluding the header and `end`.
- `arg_syms`: the argument symbols, e.g. `[:x, :y, :TV]` above.
- `args`: a modified version of the arguments changing `::Type{TV}=Vector{Float64}` and `where {TV <: AbstractVector}` to `TV::Type{<:AbstractVector}=Vector{Float64}`. This is `[:(x = zeros(2)) :y, :(TV::Type{<:AbstractVector}=Vector{Float64})]` in the example above.
- `kwargs`: the keyword arguments
- `whereparams`: the parameters in the `where` statement
and returns it as a dictionary called `model_info`.

## `translate_tilde!`

After some model information have been extracted, `translate_tilde!` replaces the `L ~ R` lines in the model with the output of `tilde(L, R, model_info)` where `L` and `R` are either expressions or symbols. In order for `tilde` to replace the `~` lines with the correct code, it checks if `L` is in `arg_syms` or not. If it isn't then `L` is definitely a parameter in `P` and the `~` line is replaced with a call to the `assume` function. If it is in the argument symbols, then it can be either in `P` or in `D` depending on whether the user provides it or not when constructing the model. This will be checked at compile time. For the variable `x` above, the `~` line is therefore replaced with:
```julia
if Turing.in_pvars(Val(:x), model)
    ...
    assume(...)
    ...
else
    ...
    observe(...)
    ...
end
```
The code before and after `assume` and `observe` as well as the arguments to both will be explained later. Which arguments have been passed in and which were missing when constructing `model::Model` is encoded in the type of `model`. `Turing.in_pvars` will therefore check if `:x` was indeed missing or not. This `if` statement will be compiled away by the Julia compiler because given the types of `Val(:x)` and `model`, one and only one branch can be executed so the Julia compiler is smart enough to optimize the condition away.

## `update_args!`

The model generator `gauss` will eventually be defined as a normal Julia function `gauss(...)`. Recall the need for a default initialization for vector variables like `x` in the model definition so when they are missing, we know their lengths. However, this is an implementation detail of what to do with missing variables, so when defining the model generator function `gauss`, it suffices to set the actual default values of the arguments to `nothing`. Inside `gauss`, we then check which arguments are `nothing` and assign them to their default initializations provided by the user. Therefore, the role of `update_args!` is to:
1. Extract the default initializations and dump them in a `NamedTuple` constructor expression called `tent_arg_defaults_nt`,
2. Ignore arguments of the form `::Type{...}` since these are treated specially (more on this later), and
3. Replace arguments and keyword arguments of the form `x = ...` with `x = nothing`.

Note that at this point arguments to the model definition and keyword arguments are combined and treated in the same way. The `NamedTuple` expression will be part of the model generator Julia function (more on this below).

## `Turing.Model`

Every `model::Model` can be called as a function with arguments:
1. `vi::VarInfo`, and
2. `spl::AbstractSampler`
`vi` is a data structure that stores information about random variables in `P`. `spl` includes the choice of the MCMC algorithm, e.g. Metropolis-Hastings, importance sampling or Hamiltonian Monte Carlo (HMC). The `assume` function will do something different for different subtypes of `AbstractSampler` to facilitate the sampling process.

The `Model` struct is defined as follows:
```julia
struct Model{pvars,
    dvars,
    F,
    TData,
    TDefaults
} <: Sampleable{VariateForm,ValueSupport} # May need to find better types
    f::F
    data::TData
    defaults::TDefaults
end
function Model{pvars, dvars}(f::F, data::TD, defaults::TDefaults) where {pvars, dvars, F, TD, TDefaults}
    return Model{pvars, dvars, F, TD, TDefaults}(f, data, defaults)
end
(model::Model)(args...; kwargs...) = model.f(args..., model; kwargs...)
```
`model.f` is an internal function that is called when `model` is called. `model` itself is passed as an argument to `model.f` because we need to access `model.data` and `model.defaults` inside `f`. `model.data` is all the variables in `D`. `model.defaults` is a `NamedTuple` of the default initializations of the missing data variables that are now part of `P`. `pvars` is the tuple of symbols of the variables in `P` and `dvars` is the tuple of symbols of the variables in `D`.

## `build_output`

Now that we have all the information we need in the `@model` macro, we can start building the model generator function. The model generator function `gauss` will be defined as:
```julia
gauss(; x = nothing, y = nothing)) = gauss(x, y)
function gauss(x = nothing, y = nothing)
    # Extract the fields of `model::Model`. `data`, `defaults` and `inner_function` will be `model.data`, `model.defaults` and `model.f`.
    
    ## `data`
    ### Finds missing inputs and adds them to `pvars` instead of `dvars`. The rest is `dvars`.
    pvars, dvars = Turing.get_vars(Tuple{:x, :y, :TV}, (x = x, y = y, TV = Vector{Float64}))
    ### Extracts `dvars` from the NamedTuple
    data = Turing.get_data(dvars, (x = x, y = y, TV = Vector{Float64}))    
    
    ## `defaults`
    ### The second argument is a `NamedTuple` of default initializations. Any default initialization of type `Array{Missing}` is replaced with an array of the same shape but of type `Array{Real}`. In the example above, `Turing.get_default_values` just returns the second argument. Note that the type parameter `TV` is NOT part of the second argument here. The second argument was defined previously in the `update_args!` section as `tent_arg_defaults_nt`.
    defaults = Turing.get_default_values(dvars, (x = zeros(2), y = nothing))

    ## `inner_function` method definitions
    inner_function(sampler::Turing.AbstractSampler, model) = inner_function(model)
    function inner_function(model)
        return inner_function(Turing.VarInfo(), Turing.SampleFromPrior(), model)
    end
    function inner_function(vi::Turing.VarInfo, model)
        return inner_function(vi, Turing.SampleFromPrior(), model)
    end
    function inner_function(vi::Turing.VarInfo, sampler::Turing.AbstractSampler, model)
        # main method
        ...
    end
    # Define and return the `model::Model` instance
    model = Turing.Model{pvars, dvars}(inner_function, data, defaults)
    return model
end
```
See the comments above for an explanation of what's happening. The body of the main method of `inner_function` is explained below.

## `inner_function`

The main method `inner_function` does some pre-processing defining all the input variables from the model definition, `x`, `y` and `TV` in the example above. Then the rest of the model body is run as normal Julia code with the `~` lines replaced with the `observe` and `assume` blocks as explained earlier.
```julia
function inner_function(vi::Turing.VarInfo, sampler::Turing.AbstractSampler, model)
    # Define the variable `x`
    local x
    # Check if `x` was given by the user when constructing `model`.
    if isdefined(model.data, :x)
        # `x` is not a type so this is a no-op
        if model.data.x isa Type && (model.data.x <: AbstractFloat || model.data.x <: AbstractArray)
            x = Turing.Core.get_matching_type(sampler, vi, model.data.x)
        else
            # Define `x` as its value provided by the user
            x = model.data.x
        end
    else
        # If `x` was not provided by the user, it is assigned to its default initialization
        x = model_defaults.x
    end

    # The same is done for `y`
    local y
    if isdefined(model.data, :y)
        if model.data.y isa Type && (model.data.y <: AbstractFloat || model.data.y <: AbstractArray)
            y = Turing.Core.get_matching_type(sampler, vi, model.data.y)
        else
            y = model.data.y
        end
    else
        y = model.defaults.y
    end

    # This is explained below in text.
    local TV
    if isdefined(model.data, :TV)
        if model.data.TV isa Type && (model.data.TV <: AbstractFloat || model.data.TV <: AbstractArray)
            TV = Turing.Core.get_matching_type(sampler, vi, model.data.TV)
        else
            TV = model.data.TV
        end
    else
        TV = model_defaults.TV
    end

    # Reset `vi.logp`, the `logpdf` of the joint probability distribution of `P` and `D`
    vi.logp = zero(Real)
    ...
end
```
As one can see from the comments in the code above, `x`, `y` and `TV` are defined in the method body followed by the rest of the code. `x`, `y` and `TV` are used in the model body; recall the model body is:
```julia
p = TV(undef, 2)
p[1] ~ InverseGamma(2, 3)
p[2] ~ Normal(0, 1.0)
for i in 1:length(x)
    x[i] ~ Normal(p[2], sqrt(p[1]))
end
y ~ Normal(p[2], sqrt(p[1]))
```

Type parameters like `TV` are treated specially though. The main purpose of the `if` statement:
```julia
if model.data.TV isa Type && (model.data.TV <: AbstractFloat || model.data.TV <: AbstractArray)
    TV = Turing.Core.get_matching_type(sampler, vi, model.data.TV)
else
    TV = model.data.TV
end
```
in the method definition above is to enable the specialization of `TV` for different `sampler` types. Some samplers such as HMC require the automatic differentiation (AD) of the `logpdf` of the joint distribution of `P` and `D`, `vi.logp`, with respect to the variables in `P`, e.g. `p` above. AD in Julia can be done using special number types such as `ForwardDiff.Dual` or `Tracker.TrackedReal`. These special number types require that the vector we are differentiating with respect to, e.g. `p`, can admit them. If `p isa Vector{Float64}`, trying to assign a `ForwardDiff.Dual` to `p[1]` will result in an error. Assume that `ForwardDiff` is used. When performing AD, the variables with respect to which we are differentiating will have a type `ForwardDiff.Dual` in `vi`. `TV = Turing.Core.get_matching_type(sampler, vi, model.data.TV)` will therefore _specialize_ `model.data.TV` replacing `Float64` in `Vector{Float64}` with `eltype(vi, sampler)` which is a concrete subtype of `ForwardDiff.Dual`. `eltype(vi, sampler)` is the number type of the variables that `sampler` is in charge of in `vi`, since `sampler` can be a `Gibbs` component in another `Gibbs` sampler. This specialization of type parameters such as `TV` enables the type-stable use of `ForwardDiff` and `Tracker` for AD without forcing the user to define `p` as a `Vector{Real}` which would cause dynamic dispatch.

# `VarName`

In order to track random variables in the sampling process, `Turing` uses the struct `VarName{sym}` which acts as a random variable identifier generated at runtime. The `VarName` of a random variable is generated from the expression on the LHS of a `~` statement when the symbol on the LHS is in `P`. Every `vn::VarName{sym}` has a symbol `sym` which is the symbol of the Julia variable in the model that the random variable belongs to. For example, `x[1] ~ Normal()` will generate an instance of `VarName{:x}` assuming `x` is in `P`. Every `vn::VarName` also has a field `indexing` which stores the indices requires to access the random variable from the Julia variable indicated by `sym`. For example, `x[1] ~ Normal()` will generate a `vn::VarName{:x}` with `vn.indexing == "[1]"`. `VarName` also supports hierarchical arrays and range indexing. Some more examples:
- `x[1] ~ Normal()` will generate a `VarName{:x}` with `indexing == "[1]"`.
- `x[:,1] ~ MvNormal(zeros(2))` will generate a `VarName{:x}` with `indexing == "[Colon(), 1]"`.
- `x[:,1][2] ~ Normal()` will generate a `VarName{:x}` with `indexing == "[Colon(), 1][2]"`.

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
Based on the type of `metadata`, the `VarInfo` is either aliased `UntypedVarInfo` or `TypedVarInfo`. `metadata` can be either a subtype of the union type `Metadata` or a `NamedTuple` of multiple such subtypes. Let `vi` be an instance of `VarInfo`. If `vi isa VarInfo{<:Metadata} == true`, then it is called an `UntypedVarInfo`. If `vi isa VarInfo{<:NamedTuple} == true`, then `vi.metadata` would be a `NamedTuple` mapping each symbol in `P` to an instance of `Metadata`. `vi` would then be called a `TypedVarInfo`. The other fields of `VarInfo` include `logp` which is used to accumulate the log probability or log probability density of the variables in `P` and `D`. `num_produce` keeps track of how many observations have been made in the model so far. This is incremented when running a `~` statement when the symbol on the LHS is in `D`.

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

# `assume` block

Recall that the `~` statements in `Turing` models get lowered in the `@model` macro to either `assume` or `observe` blocks depending on whether the symbol on the LHS of `~` is in `P` or `D` respectively. Also recall that the model's inner function wrapped in `Turing.Model` receives `vi::VarInfo` and `sampler::Sampler` as input arguments.

The `assume` block is given by:
```
varname = ...
isdist = if isa(dist, AbstractVector)
    # Check if the right-hand side is a vector of distributions.
    all(d -> isa(d, Distribution), dist)
else
    # Check if the right-hand side is a distribution.
    isa(dist, Distribution)
end
@assert isdist @error(...)

(var, lp) = if isa(dist, AbstractVector)
    Turing.assume(sampler, dist, varname, var, vi)
else
    Turing.assume(sampler, dist, varname, vi)
end
vi.logp += lp
```
where:
- `varname` is generated from the symbol or expression on the LHS of `~`,
- `dist` is everything on the RHS of the `~` statement, which is allowed to be either a distribution or a vector of distributions,
- `var` is everything on the `LHS` of the `~` statement, and
- `vi` is the `VarInfo` input to the model's inner function.

Note that `var` is replaced by whatever is on the LHS of the `~` statement, so `x[1] ~ Normal()` would expand to:
```
(x[1], lp) = if isa(dist, AbstractVector)
    Turing.assume(sampler, dist, varname, x[1], vi)
else
    Turing.assume(sampler, dist, varname, vi)
end
```
assinging the first output of the `assume` function to `x[1]` in the process.

The `assume` function can do different things for different `sampler` types modifying `vi` in the process, but whatever is done inside the specific method of the `assume` function, the value sampled and the log probability of this value are both always returned. One example of an `assume` method is:
```
function assume(spl::Sampler{<:Hamiltonian},
    dist::Distribution,
    vn::VarName,
    vi::VarInfo
)
    updategid!(vi, vn, spl)
    r = vi[vn]
    return r, logpdf_with_trans(dist, r, istrans(vi, vn))
end
```
For Hamiltonian samplers, the values of the variables are pre-loaded in `vi` before calling the model. Therefore, all the `assume` function does in this case is return the value and its transformed distribution's `logpdf`. The transformed distribution's `logpdf` of a value is obtained using `Bijectors.logpdf_with_trans` of [Bijectors.jl](https://github.com/TuringLang/Bijectors.jl).

# `observe` block

The `observe` block is simpler than the `assume` block since no instances of `VarName` are constructed for variables in `D`. The following is the `observe` block:
```
isdist = if isa(dist, AbstractVector)
    # Check if the right-hand side is a vector of distributions.
    all(d -> isa(d, Distribution), dist)
else
    # Check if the right-hand side is a distribution.
    isa(dist, Distribution)
end
@assert isdist @error(...)
vi.logp += Turing.observe(sampler, dist, observation, vi)
```
where `observation` is whatever on the LHS of `~` and `dist` is whatever on the RHS of `~`. `vi` and `sampler` are inputs to the model's inner function. Much like the `assume` function, the `observe` function is also overloaded for different samplers. The following is the `observe` method used by the importance sampling (IS) algorithm:
```
function observe(spl::Sampler{<:IS}, dist::Distribution, value::Any, vi::VarInfo)
    logpdf(dist, value)
end
```
which just returns the `logpdf` of the observation.
