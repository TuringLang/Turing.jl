module Turing

using Requires, Reexport, ForwardDiff
using DistributionsAD, Bijectors, StatsFuns, SpecialFunctions
using Statistics, LinearAlgebra
using Libtask
@reexport using Distributions, MCMCChains, Libtask, AbstractMCMC, Bijectors
using Tracker: Tracker

import AdvancedVI
using DynamicPPL: DynamicPPL, LogDensityFunction
import DynamicPPL: getspace, NoDist, NamedDist
import LogDensityProblems
import NamedArrays
import Setfield
import StatsAPI
import StatsBase

import Printf
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

include("modes/ModeEstimation.jl")
using .ModeEstimation

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
        externalsampler,

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
        LogDensityFunction,

        constrained_space,            # optimisation interface
        MAP,
        MLE,
        get_parameter_bounds,
        optim_objective,
        optim_function,
        optim_problem

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require Optim="429524aa-4258-5aef-a3af-852621145aeb" include("../ext/TuringOptimExt.jl")
    end
    @require DynamicHMC="bbc10e6e-7c05-544b-b16e-64fede858acb" begin
        @eval Inference begin
            import ..DynamicHMC
    
            if isdefined(DynamicHMC, :mcmc_with_warmup)
                include("contrib/inference/dynamichmc.jl")
            else
                error("Please update DynamicHMC, v1.x is no longer supported")
            end
        end
    end
end

end
