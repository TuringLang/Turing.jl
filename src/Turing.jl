module Turing

using Reexport, ForwardDiff
using DistributionsAD, Bijectors, StatsFuns, SpecialFunctions
using Statistics, LinearAlgebra
using Libtask
@reexport using Distributions, MCMCChains
using Compat: pkgversion

using AdvancedVI: AdvancedVI
using DynamicPPL: DynamicPPL
import DynamicPPL: getspace, NoDist, NamedDist
using LogDensityProblems: LogDensityProblems
using NamedArrays: NamedArrays
using Accessors: Accessors
using StatsAPI: StatsAPI
using StatsBase: StatsBase
using AbstractMCMC

using Accessors: Accessors

using Printf: Printf
using Random: Random

using ADTypes: ADTypes

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
    # TODO: `AdvancedVI.turnprogress` is removed in AdvancedVI v0.3
    AdvancedVI.turnprogress(progress)
    return progress
end

# Random probability measures.
include("stdlib/distributions.jl")
include("stdlib/RandomMeasures.jl")
include("essential/Essential.jl")
using .Essential
include("mcmc/Inference.jl")  # inference algorithms
using .Inference
include("variational/VariationalInference.jl")
using .Variational

include("optimisation/Optimisation.jl")
using .Optimisation

include("deprecated.jl") # to be removed in the next minor version release

###########
# Exports #
###########
# `using` statements for stuff to re-export
using DynamicPPL:
    pointwise_loglikelihoods,
    generated_quantities,
    returned,
    logprior,
    logjoint,
    condition,
    decondition,
    fix,
    unfix,
    conditioned,
    to_submodel
using StatsBase: predict
using OrderedCollections: OrderedDict

# Turing essentials - modelling macros and inference algorithms
export
    # AbstractPPL
    @varname,
    # DynamicPPL
    @model,
    to_submodel,
    @submodel,  # Deprecated in DynamicPPL
    generated_quantities, # Deprecated in DynamicPPL
    returned,
    prefix,
    NamedDist,
    predict,
    pointwise_loglikelihoods,
    logprior,
    loglikelihood,
    logjoint,
    condition,
    decondition,
    conditioned,
    fix,
    unfix,
    # OrderedCollections
    OrderedDict,
    # Turing.Inference
    setprogress!,
    Prior,
    MH,
    Emcee,
    ESS,
    Gibbs,
    HMC,
    SGLD,
    SGHMC,
    HMCDA,
    NUTS,
    PolynomialStepsize,
    IS,
    SMC,
    CSMC,
    PG,
    RepeatSampler,
    externalsampler,
    # Turing.Optimisation
    # The MAP and MLE exports are only needed for the Optim.jl interface.
    maximum_a_posteriori,
    maximum_likelihood,
    MAP,
    MLE,
    # Turing.Distributions
    Flat,
    FlatPos,
    BinomialLogit,
    OrderedLogistic,
    LogPoisson,
    # AdvancedVI
    vi,
    ADVI,
    # AbstractMCMC
    sample,
    MCMCThreads,
    MCMCDistributed,
    MCMCSerial,
    # ADTypes
    AutoForwardDiff,
    AutoReverseDiff,
    AutoZygote,
    AutoMooncake,
    # DistributionsAD
    filldist,
    arraydist

end
