module Turing

using Requires, Reexport, ForwardDiff
using DistributionsAD, Bijectors, StatsFuns, SpecialFunctions
using Statistics, LinearAlgebra
using Libtask
@reexport using Distributions, MCMCChains, Libtask, AbstractMCMC, Bijectors
using Tracker: Tracker

import AdvancedVI
using DynamicPPL: DynamicPPL
import DynamicPPL: getspace, NoDist, NamedDist
import LogDensityProblems
import Random

const PROGRESS = Ref(true)

"""
    setprogress!(progress::Bool)

Enable progress logging in Turing if `progress` is `true`, and disable it otherwise.
"""
function setprogress!(progress::Bool)
    @info "[Turing]: progress logging is $(progress ? "enabled" : "disabled") globally"
    PROGRESS[] = progress
    AdvancedVI.turnprogress(progress)
    return progress
end

# Log density function
struct LogDensityFunction{V,M,C}
    varinfo::V
    model::M
    context::C
end

function LogDensityFunction(
    varinfo::DynamicPPL.AbstractVarInfo,
    model::DynamicPPL.Model,
    sampler::DynamicPPL.AbstractSampler,
    context::DynamicPPL.AbstractContext
)
    return LogDensityFunction(varinfo, model, DynamicPPL.SamplingContext(sampler, context))
end

# Convenient for end-user.
function LogDensityFunction(
    model::DynamicPPL.Model,
    varinfo::DynamicPPL.AbstractVarInfo=DynamicPPL.VarInfo(model),
    context::DynamicPPL.AbstractContext=DynamicPPL.DefaultContext(),
)
    return LogDensityFunction(model, varinfo, context)
end

# HACK: heavy usage of `AbstractSampler` for, well, _everything_, is being phased out. In the mean time
# we need to define these annoying methods to ensure that we stay compatible with everything.
getsampler(f::LogDensityFunction) = getsampler(f.context)
getsampler(ctx::DynamicPPL.SamplingContext) = ctx.sampler
getsampler(ctx::DynamicPPL.AbstractContext) = getsampler(DynamicPPL.NodeTrait(ctx), ctx)
getsampler(::DynamicPPL.IsParent, ctx::DynamicPPL.AbstractContext) = getsampler(DynamicPPL.childcontext(ctx))

hassampler(f::LogDensityFunction) = hassampler(f.context)
hassampler(ctx::DynamicPPL.SamplingContext) = true
hassampler(ctx::DynamicPPL.AbstractContext) = hassampler(DynamicPPL.NodeTrait(ctx), ctx)
hassampler(::DynamicPPL.IsLeaf, ctx::DynamicPPL.AbstractContext) = false
hassampler(::DynamicPPL.IsParent, ctx::DynamicPPL.AbstractContext) = hassampler(DynamicPPL.childcontext(ctx))

# TODO: This is annoying. Should all this be moved to DynamicPPL so we can share the impl of `unflatten`?
function unflatten(varinfo, context, θ)
    return hassampler(context) ? unflatten(getsampler(context), varinfo, context, θ) : DynamicPPL.unflatten(varinfo, θ)
end
unflatten(sampler::DynamicPPL.AbstractSampler, varinfo, context, θ) = DynamicPPL.unflatten(varinfo, sampler, θ)

function (f::LogDensityFunction)(θ::AbstractVector)
    vi_new = unflatten(f.varinfo, f.context, θ)
    return getlogp(last(DynamicPPL.evaluate!!(f.model, vi_new, f.context)))
end

# LogDensityProblems interface
LogDensityProblems.logdensity(f::LogDensityFunction, θ) = f(θ)
function LogDensityProblems.capabilities(::Type{<:LogDensityFunction})
    return LogDensityProblems.LogDensityOrder{0}()
end

_get_indexer(ctx::DynamicPPL.AbstractContext) = _get_indexer(DynamicPPL.NodeTrait(ctx), ctx)
_get_indexer(ctx::DynamicPPL.SamplingContext) = ctx.sampler
function _get_indexer(::DynamicPPL.IsParent, ctx::DynamicPPL.AbstractContext)
    return _get_indexer(DynamicPPL.childcontext(ctx))
end
function _get_indexer(::DynamicPPL.IsLeaf, ctx::DynamicPPL.AbstractContext)
    return Colon()
end
getparams(f::LogDensityFunction) = f.varinfo[_get_indexer(f.context)]

LogDensityProblems.dimension(f::LogDensityFunction) = length(getparams(f))

# Standard tag: Improves stacktraces
# Ref: https://www.stochasticlifestyle.com/improved-forwarddiff-jl-stacktraces-with-package-tags/
struct TuringTag end

# Allow Turing tag in gradient etc. calls of the log density function
ForwardDiff.checktag(::Type{ForwardDiff.Tag{TuringTag, V}}, ::LogDensityFunction, ::AbstractArray{V}) where {V} = true
ForwardDiff.checktag(::Type{ForwardDiff.Tag{TuringTag, V}}, ::Base.Fix1{typeof(LogDensityProblems.logdensity),<:LogDensityFunction}, ::AbstractArray{V}) where {V} = true

# Random probability measures.
include("stdlib/distributions.jl")
include("stdlib/RandomMeasures.jl")
include("utilities/Utilities.jl")
using .Utilities
include("essential/Essential.jl")
Base.@deprecate_binding Core Essential false
using .Essential
include("inference/Inference.jl")  # inference algorithms
using .Inference
include("variational/VariationalInference.jl")
using .Variational

@init @require DynamicHMC="bbc10e6e-7c05-544b-b16e-64fede858acb" begin
    @eval Inference begin
        import ..DynamicHMC

        if isdefined(DynamicHMC, :mcmc_with_warmup)
            include("contrib/inference/dynamichmc.jl")
        else
            error("Please update DynamicHMC, v1.x is no longer supported")
        end
    end
end

include("modes/ModeEstimation.jl")
using .ModeEstimation

@init @require Optim="429524aa-4258-5aef-a3af-852621145aeb" @eval begin
    include("modes/OptimInterface.jl")
    export optimize
end

###########
# Exports #
###########
# `using` statements for stuff to re-export
using DynamicPPL: pointwise_loglikelihoods, generated_quantities, logprior, logjoint
using StatsBase: predict

# Turing essentials - modelling macros and inference algorithms
export  @model,                 # modelling
        @varname,
        @submodel,
        DynamicPPL,

        Prior,                  # Sampling from the prior

        MH,                     # classic sampling
        RWMH,
        Emcee,
        ESS,
        Gibbs,
        GibbsConditional,

        HMC,                    # Hamiltonian-like sampling
        SGLD,
        SGHMC,
        HMCDA,
        NUTS,
        DynamicNUTS,
        ANUTS,

        PolynomialStepsize,

        IS,                     # particle-based sampling
        SMC,
        CSMC,
        PG,

        vi,                     # variational inference
        ADVI,

        sample,                 # inference
        resume,
        @logprob_str,
        @prob_str,

        setchunksize,           # helper
        setadbackend,
        setadsafe,

        setprogress!,           # debugging

        Flat,
        FlatPos,
        BinomialLogit,
        BernoulliLogit,         # Part of Distributions >= 0.25.77
        OrderedLogistic,
        LogPoisson,
        NamedDist,
        filldist,
        arraydist,

        predict,
        pointwise_loglikelihoods,
        elementwise_loglikelihoods,
        generated_quantities,
        logprior,
        logjoint,

        constrained_space,            # optimisation interface
        MAP,
        MLE,
        get_parameter_bounds,
        optim_objective,
        optim_function,
        optim_problem
end
