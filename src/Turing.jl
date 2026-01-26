module Turing

using Reexport, ForwardDiff
using DistributionsAD, Bijectors, StatsFuns, SpecialFunctions
using Statistics, LinearAlgebra
using Libtask
@reexport using Distributions, MCMCChains
using Compat: pkgversion

using AdvancedVI: AdvancedVI
using DynamicPPL: DynamicPPL
import DynamicPPL: NoDist, NamedDist
using LogDensityProblems: LogDensityProblems
using Accessors: Accessors
using StatsAPI: StatsAPI
using StatsBase: StatsBase
using AbstractMCMC

using Accessors: Accessors

using Printf: Printf
using Random: Random
using LinearAlgebra: I

using ADTypes: ADTypes, AutoForwardDiff, AutoReverseDiff, AutoMooncake, AutoEnzyme

const DEFAULT_ADTYPE = ADTypes.AutoForwardDiff()

const PROGRESS = Ref(true)

# TODO: remove `PROGRESS` and this function in favour of `AbstractMCMC.PROGRESS`
"""
    setprogress!(progress::Bool)

Enable progress logging in Turing if `progress` is `true`, and disable it otherwise.
"""
function setprogress!(progress::Bool)
    @info "[Turing]: progress logging is $(progress ? "enabled" : "disabled") globally"
    PROGRESS[] = progress
    AbstractMCMC.setprogress!(progress; silent=true)
    return progress
end

# Random probability measures.
include("stdlib/distributions.jl")
include("stdlib/RandomMeasures.jl")
include("init_strategy.jl")
include("mcmc/Inference.jl")  # inference algorithms
using .Inference
include("variational/Variational.jl")
using .Variational

include("optimisation/Optimisation.jl")
using .Optimisation

###########
# Exports #
###########
# `using` statements for stuff to re-export
using DynamicPPL:
    @model,
    @varname,
    pointwise_loglikelihoods,
    generated_quantities,
    returned,
    logprior,
    logjoint,
    condition,
    decondition,
    fix,
    unfix,
    prefix,
    conditioned,
    to_submodel,
    LogDensityFunction,
    @addlogprob!,
    InitFromPrior,
    InitFromUniform,
    InitFromParams,
    setthreadsafe
using StatsBase: predict
using OrderedCollections: OrderedDict

# Turing essentials - modelling macros and inference algorithms
export
    # DEPRECATED
    generated_quantities,
    # Modelling - AbstractPPL and DynamicPPL
    @model,
    @varname,
    to_submodel,
    prefix,
    LogDensityFunction,
    @addlogprob!,
    setthreadsafe,
    # Sampling - AbstractMCMC
    sample,
    MCMCThreads,
    MCMCDistributed,
    MCMCSerial,
    # Samplers - Turing.Inference
    Prior,
    MH,
    Emcee,
    ESS,
    Gibbs,
    GibbsConditional,
    HMC,
    SGLD,
    SGHMC,
    PolynomialStepsize,
    HMCDA,
    NUTS,
    IS,
    SMC,
    PG,
    CSMC,
    RepeatSampler,
    externalsampler,
    # Variational inference - AdvancedVI
    vi,
    q_locationscale,
    q_meanfield_gaussian,
    q_fullrank_gaussian,
    KLMinRepGradProxDescent,
    KLMinRepGradDescent,
    KLMinScoreGradDescent,
    KLMinNaturalGradDescent,
    KLMinSqrtNaturalGradDescent,
    KLMinWassFwdBwd,
    FisherMinBatchMatch,
    # ADTypes
    AutoForwardDiff,
    AutoReverseDiff,
    AutoMooncake,
    AutoEnzyme,
    # Debugging - Turing
    setprogress!,
    # Distributions
    Flat,
    FlatPos,
    BinomialLogit,
    OrderedLogistic,
    LogPoisson,
    # Tools to work with Distributions
    I,  # LinearAlgebra
    filldist,  # DistributionsAD
    arraydist,  # DistributionsAD
    NamedDist,  # DynamicPPL
    # Predictions - DynamicPPL
    predict,
    # Querying model probabilities - DynamicPPL
    returned,
    pointwise_loglikelihoods,
    logprior,
    loglikelihood,
    logjoint,
    condition,
    decondition,
    conditioned,
    fix,
    unfix,
    OrderedDict, # OrderedCollections
    # Initialisation strategies for models
    InitFromPrior,
    InitFromUniform,
    InitFromParams,
    # Point estimates - Turing.Optimisation
    # The MAP and MLE exports are only needed for the Optim.jl interface.
    maximum_a_posteriori,
    maximum_likelihood,
    MAP,
    MLE,
    get_vector_params,
    # Chain save/resume
    loadstate

end
