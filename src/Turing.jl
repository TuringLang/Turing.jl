module Turing

##############
# Dependency #
########################################################################
# NOTE: when using anything from external packages,                    #
#       let's keep the practice of explictly writing Package.something #
#       to indicate that's not implemented inside Turing.jl            #
########################################################################

using Requires, Reexport, ForwardDiff
using Bijectors, StatsFuns, SpecialFunctions
using Statistics, LinearAlgebra, ProgressMeter
using Markdown, Libtask, MacroTools
using AbstractMCMC
@reexport using Distributions, MCMCChains, Libtask
using Tracker: Tracker

import Base: ~, ==, convert, hash, promote_rule, rand, getindex, setindex!
import MCMCChains: AbstractChains, Chains

const PROGRESS = Ref(true)
function turnprogress(switch::Bool)
    @info("[Turing]: global PROGRESS is set as $switch")
    PROGRESS[] = switch
end

# Constants for caching
const CACHERESET  = 0b00
const CACHEIDCS   = 0b10
const CACHERANGES = 0b01

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_TURING", "0")))

# Random probability measures.
include("stdlib/distributions.jl")
include("stdlib/RandomMeasures.jl")

"""
    struct Model{F, Targs <: NamedTuple, Tmodelgen, Tmissings <: Val}
        f::F
        args::Targs
        modelgen::Tmodelgen
        missings::Tmissings
    end

A `Model` struct with arguments `args`, inner function `f`, model generator `modelgen` and
missing data `missings`. `missings` is a `Val` instance, e.g. `Val{(:a, :b)}()`. An
argument in `args` with a value `missing` will be in `missings` by default. However, in
non-traditional use-cases `missings` can be defined differently. All variables in
`missings` are treated as random variables rather than observations.
"""
struct Model{F, Targs <: NamedTuple, Tmodelgen, Tmissings <: Val} <: AbstractModel
    f::F
    args::Targs
    modelgen::Tmodelgen
    missings::Tmissings
end
Model(f, args::NamedTuple, modelgen) = Model(f, args, modelgen, getmissing(args))
(model::Model)(vi) = model(vi, SampleFromPrior())
(model::Model)(vi, spl) = model(vi, spl, DefaultContext())
(model::Model)(args...; kwargs...) = model.f(args..., model; kwargs...)

getmissing(model::Model) = model.missings
@generated function getmissing(args::NamedTuple{names, ttuple}) where {names, ttuple}
    length(names) == 0 && return :(Val{()}())
    minds = filter(1:length(names)) do i
        ttuple.types[i] == Missing
    end
    mnames = names[minds]
    return :(Val{$mnames}())
end

function runmodel! end
function getspace end

struct Selector
    gid :: UInt64
    tag :: Symbol # :default, :invalid, :Gibbs, :HMC, etc.
end
Selector() = Selector(time_ns(), :default)
Selector(tag::Symbol) = Selector(time_ns(), tag)
hash(s::Selector) = hash(s.gid)
==(s1::Selector, s2::Selector) = s1.gid == s2.gid

"""
Robust initialization method for model parameters in Hamiltonian samplers.
"""
struct SampleFromUniform <: AbstractSampler end
struct SampleFromPrior <: AbstractSampler end

getspace(::Union{SampleFromPrior, SampleFromUniform}) = ()

"""
An abstract type that mutable sampler state structs inherit from.
"""
abstract type AbstractSamplerState end

"""
    Sampler{T}

Generic interface for implementing inference algorithms.
An implementation of an algorithm should include the following:

1. A type specifying the algorithm and its parameters, derived from InferenceAlgorithm
2. A method of `sample` function that produces results of inference, which is where actual inference happens.

Turing translates models to chunks that call the modelling functions at specified points.
The dispatch is based on the value of a `sampler` variable.
To include a new inference algorithm implements the requirements mentioned above in a separate file,
then include that file at the end of this one.
"""
mutable struct Sampler{T, S<:AbstractSamplerState} <: AbstractSampler
    alg      ::  T
    info     ::  Dict{Symbol, Any} # sampler infomation
    selector ::  Selector
    state    ::  S
end
Sampler(alg) = Sampler(alg, Selector())
Sampler(alg, model::Model) = Sampler(alg, model, Selector())
Sampler(alg, model::Model, s::Selector) = Sampler(alg, model, s)

abstract type AbstractContext end

"""
    struct DefaultContext <: AbstractContext end

The `DefaultContext` is used by default to compute log the joint probability of the data 
and parameters when running the model.
"""
struct DefaultContext <: AbstractContext end

"""
    struct PriorContext{Tvars} <: AbstractContext
        vars::Tvars
    end

The `PriorContext` enables the computation of the log prior of the parameters `vars` when 
running the model.
"""
struct PriorContext{Tvars} <: AbstractContext
    vars::Tvars
end
PriorContext() = PriorContext(nothing)

"""
    struct LikelihoodContext{Tvars} <: AbstractContext
        vars::Tvars
    end

The `LikelihoodContext` enables the computation of the log likelihood of the data when 
running the model. `vars` can be used to evaluate the log likelihood for specific values 
of the model's parameters. If `vars` is `nothing`, the parameter values inside the `VarInfo` will be used by default.
"""
struct LikelihoodContext{Tvars} <: AbstractContext
    vars::Tvars
end
LikelihoodContext() = LikelihoodContext(nothing)

"""
    struct MiniBatchContext{Tctx, T} <: AbstractContext
        ctx::Tctx
        loglike_scalar::T
    end

The `MiniBatchContext` enables the computation of 
`log(prior) + s * log(likelihood of a batch)` when running the model, where `s` is the 
`loglike_scalar` field, typically equal to `the number of data points / batch size`. 
This is useful in batch-based stochastic gradient descent algorithms to be optimizing 
`log(prior) + log(likelihood of all the data points)` in the expectation.
"""
struct MiniBatchContext{Tctx, T} <: AbstractContext
    ctx::Tctx
    loglike_scalar::T
end
function MiniBatchContext(ctx = DefaultContext(); batch_size, npoints)
    return MiniBatchContext(ctx, npoints/batch_size)
end
include("utilities/Utilities.jl")
using .Utilities
include("core/Core.jl")
using .Core
include("inference/Inference.jl")  # inference algorithms
using .Inference
include("variational/VariationalInference.jl")
using .Variational

# TODO: re-design `sample` interface in MCMCChains, which unify CmdStan and Turing.
#   Related: https://github.com/TuringLang/Turing.jl/issues/746
#@init @require CmdStan="593b3428-ca2f-500c-ae53-031589ec8ddd" @eval begin
#     @eval Utilities begin
#         using ..Turing.CmdStan: CmdStan, Adapt, Hmc
#         using ..Turing: HMC, HMCDA, NUTS
#         include("utilities/stan-interface.jl")
#     end
# end

@init @require DynamicHMC="bbc10e6e-7c05-544b-b16e-64fede858acb" @eval Inference begin
    using Pkg; 
    Pkg.installed()["DynamicHMC"] < v"2.0" && error("Please upgdate your DynamicHMC, v1.x is no longer supported")
    using ..Turing.DynamicHMC: DynamicHMC, mcmc_with_warmup
    include("contrib/inference/dynamichmc.jl")
end

###########
# Exports #
###########

# Turing essentials - modelling macros and inference algorithms
export  @model,                 # modelling
        @varname,
        @varinfo,
        @logpdf,
        @sampler,

        MH,                     # classic sampling
        Gibbs,

        HMC,                    # Hamiltonian-like sampling
        SGLD,
        SGHMC,
        HMCDA,
        NUTS,
        DynamicNUTS,
        ANUTS,

        IS,                     # particle-based sampling
        SMC,
        CSMC,
        PG,
        PIMH,
        PMMH,
        IPMCMC,

        vi,                    # variational inference
        ADVI,

        sample,                 # inference
        psample,
        setchunksize,
        resume,
        @logprob_str,
        @prob_str,

        auto_tune_chunk_size!,  # helper
        setadbackend,
        setadsafe,

        turnprogress,           # debugging

        Flat,
        FlatPos,
        BinomialLogit,
        VecBinomialLogit,
        OrderedLogistic,
        LogPoisson,
        NamedDist

end
