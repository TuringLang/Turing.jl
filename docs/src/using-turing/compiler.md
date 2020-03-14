---
title: Turing Compiler Design
---

In this section, I will describe the current design of Turing's model compiler which enables Turing to perform various types of Bayesian inference without changing the model definition. What we call "compiler" is essentially just a macro that transforms the user's code to something that Julia's dispatch can operate on and that Julia's compiler can successfully do type inference on for efficient machine code generation.

# Overview

The following terminology will be used in this section:

- `D`: observed data variables conditioned upon in the posterior,
- `P`: parameter variables distributed according to the prior distributions, these will also be referred to as random variables,
- `Model`: a fully defined probabilistic model with input data, and
- `ModelGen`: a model generator function that can be used to instantiate a `Model` instance by inputing data `D`.

`Turing`'s `@model` macro defines a `ModelGen` that can be used to instantiate a `Model` by passing in the observed data `D`.

# `@model` macro and `ModelGen`

The following are the main jobs of the `@model` macro:
1. Parse `~` and `.~` lines, e.g. `y .~ Normal.(c*x, 1.0)`
2. Figure out if a variable belongs to the data `D` and or to the parameters `P`
3. Enable the handling of missing data variables in `D` when defining a `Model` and treating them as parameter variables in `P` instead
4. Enable the tracking of random variables using the data structures `VarName` and `VarInfo`
5. Change `~`/`.~` lines with a variable in `P` on the LHS to a call to an `assume`/`dot_assume`-block
6. Change `~`/`.~` lines with a variable in `D` on the LHS to a call to an `observe`/`dot_observe`-block
7. Enable type stable automatic differentiation of the model using type parameters

Let's take the following model as an example:
```julia
@model gauss(x = missing, y = 1.0, ::Type{TV} = Vector{Float64}) where {TV <: AbstractVector} = begin
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
The above call of the `@model` macro defines an instance of `ModelGen` called `gauss`. A `model::Model` can be defined using `gauss(rand(3), 1.0)` or `gauss(x = rand(3), y = 1.0)`. While constructing the model, if an argument is not passed in, it gets assigned to its default value. If there is no default value given, an error is thrown. If an argument has a default value `missing`, when not passed in, it is treated as a random variable. For variables which require an intialization because we need to loop or broadcast over its elements, such as `x` above, the following needs to be done:
```julia
if x === missing
    x = ...
end
```
If `x` is sampled as a whole from a multivariate distribution, e.g. `x ~ MvNormal(...)`, there is no need to initialize it in an `if`-block.

`ModelGen` is defined as:
```julia
struct ModelGen{Targs, F, Tdefaults} <: Function
    f::F
    defaults::Tdefaults
end
ModelGen{Targs}(args...) where {Targs} = ModelGen{Targs, typeof.(args)...}(args...)
(m::ModelGen)(args...; kwargs...) = m.f(args...; kwargs...)
```
`Targs` is the tuple of the symbols of the model's arguments, `(:x, :y, :TV)`. `defaults` is the `NamedTuple` of default values `(x = missing, y = 1.0, TV = Vector{Float64})`.

The `@model` macro is defined as:
```julia
macro model(input_expr)
    build_model_info(input_expr) |> replace_tilde! |> replace_vi! |> 
        replace_logpdf! |> replace_sampler! |> build_output
end
```

## `build_model_info`

The first stop that the model definition takes is `build_model_info`. This function extracts some information from the model definition such as:
- `name`: the model name.
- `main_body`: the model body excluding the header and `end`.
- `arg_syms`: the argument symbols, e.g. `[:x, :y, :TV]` above.
- `args`: a modified version of the arguments changing `::Type{TV}=Vector{Float64}` and `where {TV <: AbstractVector}` to `TV::Type{<:AbstractVector}=Vector{Float64}`. This is `[:(x = missing) :(y = 1.0), :(TV::Type{<:AbstractVector}=Vector{Float64})]` in the example above.
- `args_nt`: an expression constructing a `NamedTuple` of the input arguments, e.g. :((x = x, y = y, TV = TV)) in the example above.
- `defaults_nt`: an expression constructing a `NamedTuple` of the default values of the input arguments, if any, e.g. :((x = missing, y = 1, TV = Vector{Float64})) in the example above.
and returns it as a dictionary called `model_info`.

## `replace_tilde!`

After some model information have been extracted, `replace_tilde!` replaces the `L ~ R` lines in the model with the output of `Core.tilde(L, R, model_info)` where `L` and `R` are either expressions or symbols. `L` can also be a constant literal. The `replace_tilde!` function also replaces expressions of the form `@. L ~ R` with the output of `dot_tilde(L, R, model_info)`.

In the above example, `p[1] ~ InverseGamma(2, 3)` is replaced with:
```julia
temp_right = InverseGamma(2, 3)
Turing.Core.assert_dist(temp_right, msg = ...)
preprocessed = Turing.Core.@preprocess(Val((:x, :y, :T)), Turing.getmissing(model), p[1])
if preprocessed isa Tuple
    vn, inds = preprocessed
    out = Turing.Inference.tilde(ctx, sampler, temp_right, vn, inds, vi)
    p[1] = out[1]
    acclogp!(vi, out[2])
else
    acclogp!(vi, Turing.Inference.tilde(ctx, sampler, temp_right, preprocessed, vi))
end
```
where `ctx::AbstractContext`, `sampler::AbstractSampler` and `vi::VarInfo` will be discussed later. `assert_dist` will check that the RHS of `~` is a distribution otherwise an error is thrown. The `@preprocess` macro here checks:
1. If the symbol on the LHS of `~`, `:p` in this case, is in the arguments to the model, `(:x, :y, :T)`, or not. If it isn't, then `p[1]` will be treated as a random variable. 
2. If it is in the arguments but was among the arguments with a value of `missing`, obtained using `getmissing(model)`, then `p[1]` is also treated as a random variable.
3. If neither of the above is true, but the value of `p[1]` is `missing`, then `p[1]` will still be treated as a random variable.
4. Otherwise, `p[1]` is treated as an observation.

If `@preprocess` treats `p[1]` as a random variable, it will return a `2-Tuple` of: 1) a variable identifier `vn::VarName = Turing.@varname p[1]`, and 2) a tuple of tuples of the indices used in `vn`, `((1,),)` in this example. Otherwise, `@preprocess` returns the value of `p[1]`. `Turing.@varname` and `VarName` wil be explained later. The above checks by `@preprocess` were carefully written to make sure that the Julia compiler can compile them away so no checks happen at runtime and only the correct branch is run straight away. 

When the output of `@preprocess` is a `Tuple`, i.e. `p[1]` is a random variable, the `Turing.Inference.tilde` function will dispatch to a different method than when the output is of another type, i.e `p[1]` is an observation. In the former case, `Turing.Inference.tilde` returns 2 outputs, the value of the random variable and the `log` probability, while in the latter case, only the `log` probability is returned. The `log` probabilities then get accumulated and if `p[1]` is a random variable, the first returned output by `Turing.Inference.tilde` gets assigned to it.

Note that `Core.tilde` is different from `Inference.tilde`. `Core.tilde` returns the expression block that will be run instead of the `~` line. A part of this expression block is a call to `Inference.tilde` as shown above. `Core.tilde` is defined in the `compiler.jl` file, while `Inference.tilde` is defined in the `Inference.jl` file.

The `dot_tilde!` function does something similar for expressions of the form `@. L ~ R` (and `L .~ R` in Julia 1.1 and above). Let's take `@. x[1:2] ~ Normal(p[2], sqrt(p[1]))` as an example. This expressions replaced with:
```julia
temp_right = Normal(p[2], sqrt(p[1]))
Turing.Core.assert_dist(temp_right, msg = ...)
preprocessed = Turing.Core.@preprocess(Val((:x, :y, :T)), Turing.getmissing(model), x[1:2])
if preprocessed isa Tuple
    vn, inds = preprocessed
    temp_left = x[1:2]
    out = Turing.Inference.dot_tilde(ctx, sampler, temp_right, temp_left, vn, inds, vi)
    left .= out[1]
    acclogp!(vi, out[2])
else
    temp_left = preprocessed # x[1:2]
    acclogp!(vi, Turing.Inference.dot_tilde(ctx, sampler, temp_right, temp_left, vi))
end
```
The main difference in the expanded code between `L ~ R` and `@. L ~ R` is that the former doesn't assume `L` to be defined, it can be a new Julia variable in the scope, while the latter assumes `L` already exists. `L` is also always input to the `dot_tilde` function but not the `tilde` function.

## `replace_vi!`, `replace_logpdf!` and `replace_sampler!`

Using `@varinfo()` inside the model body will give the user access to the `vi::VarInfo` object used inside the model. The function `replace_vi!` therefore finds and replaces every use of `@varinfo()` with the handle to the `VarInfo` instance used inside the model. The `@logpdf()` macro will return `vi.logp[]` which is the accumumlated `log` probability that the model is computing. What this means can change depending on the context, `ctx`, used when running the model. Finally, `replace_sampler!` will replace `@sampler()` with the `sampler` input to the model.

## `Turing.Model`

Every `model::Model` can be called as a function with arguments:
1. `vi::VarInfo`,
2. `spl::AbstractSampler`, and
3. `ctx::AbstractContext`.
`vi` is a data structure that stores information about random variables in `P`. `spl` includes the choice of the MCMC algorithm, e.g. Metropolis-Hastings, importance sampling or Hamiltonian Monte Carlo (HMC). `ctx` is used to modify the behaviour of the `logp` accumulator, accumulating different variants of it. For example, if `ctx isa LikelihoodContext`, only the log likelihood will be accumulated in `vi.logp[]`. By default, `ctx isa DefaultContext` which accumulates the log joint probability of `P` and `D`. The `Inference.tilde` and `Inference.dot_tilde` functions will do something different for different subtypes of `AbstractSampler` to facilitate the sampling process.

The `Model` struct is defined as follows:
```julia
struct Model{F, Targs <: NamedTuple, Tmodelgen, Tmissings <: Val}
    f::F
    args::Targs
    modelgen::Tmodelgen
    missings::Tmissings
end
Model(f, args::NamedTuple, modelgen) = Model(f, args, modelgen, getmissing(args))
(model::Model)(vi) = model(vi, SampleFromPrior())
(model::Model)(vi, spl) = model(vi, spl, DefaultContext())
(model::Model)(args...; kwargs...) = model.f(args..., model; kwargs...)
```
`model.f` is an internal function that is called when `model` is called, where `model::Model`. When `model` is called, `model` itself is passed as an argument to `model.f` because we need to access `model.args` among other things inside `f`. `model.args` is a `NamedTuple` of all the arguments that were passed to the model generating function when constructing an instance of `Model`. `modelgen` is the instance of `ModelGen` that was used to construct `model`. `missings` is an instance of `Val`, e.g. `Val{(:a, :b)}()`. `getmissings` returns a `Val` instance of all the symbols in `args` with a value `missing`. This is the default definition of `missings`. All variables in `missings` are treated as random variables rather than observations. 

In some non-traditional use-cases, `missings` is defined differently, e.g. when computing the log joint probability of the random variables and only some observations simultaneously, possibly conditioned on the remaining observations. An example using the model above is `logprob"x = rand(3), p = rand(2) | model = gauss, y = nothing"`. To evaluate this, the model argument `x` on the LHS of `|` is treated as a random variable leading to a call to the `assume` or `dot_assume` function in place of the `~` or `.~` expressions, respectively. The model is then run in the `PriorContext` which ignores the `observe` and `dot_observe` functions and only runs the `assume` and `dot_assume` ones. This returns the correct log probability. The reason why a model input argument, such as `x`, cannot be initialized to `missing` when on the LHS of `|` is somewhat subtle. In the model body before calling `~`, sometimes there would be a call to `length(x)` iterating over the elements of `x` in a loop calling `~` on each element of `x`. If `x` is initialized to `missing`, this will error because `length(missing)` is not defined. Moreover, it is not intuitive to require the user to handle the `x === missing` case because the user never assigned `x` to be `missing` in the first place, `missing` is merely an implementation detail in this case that the users need not concern themselves with. Therefore in this case, it makes sense to de-couple the `missings` field from the values of the arguments.

## `build_output`

Now that we have all the information we need in the `@model` macro, we can start building the model generator function. The model generator function `gauss` will be defined as:
```julia
function outer_function(;
    x = missing,
    y = 1.0,
    TV::Type{<:AbstractVector} = Vector{Float64},
)
    return outer_function(x, y, TV)
end
function outer_function(
    x = missing,
    y = 1.0,
    TV::Type{<:AbstractVector} = Vector{Float64},
)
    function inner_function(vi::Turing.VarInfo, sampler::Turing.AbstractSampler, ctx::AbstractContext, model)
        ...
    end
    return Turing.Model(
        inner_function,
        (x = x, y = y, TV = TV),
        Turing.Core.ModelGen{(:x, :y, :TV)}(
            outer_function,
            (x = missing, y = 1.0, TV = Vector{Float64}),
        ),
    )
end
gauss = Turing.Core.ModelGen{(:x, :y, :TV)}(
            outer_function,
            (x = missing, y = 1.0, TV = Vector{Float64}),
        )
```
The above 2 methods enable constructing the model using positional or keyword arguments. The second argument to the `Turing.Model` constructor is the expression called `args_nt` stored in `model_info`. The second argument to the `ModelGen` constructor inside `outer_function` and outside is the expression called `defaults_nt` stored in `model_info`. The body of the `inner_function` is explained below.

## `inner_function`

The main method of `inner_function` does some pre-processing defining all the input variables from the model definition, `x`, `y` and `TV` in the example above. Then the rest of the model body is run as normal Julia code with the `L ~ R` and `@. L ~ R` lines replaced with the calls to `Inference.tilde` and `Inference.dot_tilde` respectively as shown earlier.
```julia
function inner_function(vi::Turing.VarInfo, sampler::Turing.AbstractSampler, ctx::AbstractContext, model)
    temp_x = model.args.x
    xT = typeof(temp_x)
    if temp_x isa Turing.Core.FloatOrArrayType
        x = Turing.Core.get_matching_type(sampler, vi, temp_x)
    elseif Turing.Core.hasmissing(xT)
        x = Turing.Core.get_matching_type(sampler, vi, xT)(temp_x)
    else
        x = temp_x
    end

    ... # The code above is repeated for the other 2 variables, y and TV

    # Reset the `logp` accumulator
    resetlogp!(vi)

    ... # Main model body
end
```
As one can see above, `x`, `y` and `TV` are defined in the method body using an `if`-block followed by the rest of the code. The first branch of this `if`-block is run if the variable is a number or array type, such as `TV = Vector{Float64}`. One of the purposes of `get_matching_type` is to check if `sampler` requires automatic differentiation, and to modify `TV` accordingly. For example, when using `ForwardDiff` for automatic differentiation, `TV` will be defined as some concrete subtype of `Vector{<:ForwardDiff.Dual}`. This same function is also used to replace `Array` with `Libtask.TArray` types when a particle sampler is used.

The second branch of the `if`-block is to handle partially missing data converting the type of the input vector to another type befitting of the sampler used, whether it is for automatic differentiation or for particle samplers. Finally, the third branch is the one that will be run for `x` and `y` above simply assigning these names to `model.args.x` and `model.args.y` respectively. The main model body is then the same model body passed in by the user after replacing `L ~ R`, `@. L ~ R`, `@varinfo()` and `@logpdf()` as explained eariler.

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
