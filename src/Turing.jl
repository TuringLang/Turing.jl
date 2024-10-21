module Turing

using Reexport, ForwardDiff
using DistributionsAD, Bijectors, StatsFuns, SpecialFunctions
using Statistics, LinearAlgebra
using Libtask
@reexport using Distributions, MCMCChains, Libtask, AbstractMCMC, Bijectors
using Compat: pkgversion

using AdvancedVI: AdvancedVI
using DynamicPPL: DynamicPPL, LogDensityFunction
import DynamicPPL: getspace, NoDist, NamedDist
using LogDensityProblems: LogDensityProblems
using NamedArrays: NamedArrays
using Accessors: Accessors
using StatsAPI: StatsAPI
using StatsBase: StatsBase

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

include("experimental/Experimental.jl")
include("deprecated.jl") # to be removed in the next minor version release

###########
# Exports #
###########
# `using` statements for stuff to re-export
using DynamicPPL:
    pointwise_loglikelihoods,
    generated_quantities,
    logprior,
    logjoint,
    condition,
    decondition,
    fix,
    unfix,
    conditioned
using StatsBase: predict
using Bijectors: ordered
using OrderedCollections: OrderedDict

# Turing essentials - modelling macros and inference algorithms
export @model,                 # modelling
    @varname,
    @submodel,
    DynamicPPL,
    Prior,                  # Sampling from the prior
    MH,                     # classic sampling
    Emcee,
    ESS,
    Gibbs,
    GibbsConditional,
    HMC,                    # Hamiltonian-like sampling
    SGLD,
    SGHMC,
    HMCDA,
    NUTS,
    PolynomialStepsize,
    IS,                     # particle-based sampling
    SMC,
    CSMC,
    PG,
    vi,                     # variational inference
    ADVI,
    sample,                 # inference
    @logprob_str,  # TODO: Remove, see https://github.com/TuringLang/DynamicPPL.jl/issues/356
    @prob_str,     # TODO: Remove, see https://github.com/TuringLang/DynamicPPL.jl/issues/356
    externalsampler,
    AutoForwardDiff,        # ADTypes
    AutoReverseDiff,
    AutoZygote,
    AutoMooncake,
    setprogress!,           # debugging
    Flat,
    FlatPos,
    BinomialLogit,
    BernoulliLogit,         # Part of Distributions >= 0.25.77
    OrderedLogistic,
    LogPoisson,
    filldist,
    arraydist,
    NamedDist,              # Exports from DynamicPPL
    predict,
    pointwise_loglikelihoods,
    generated_quantities,
    logprior,
    logjoint,
    LogDensityFunction,
    condition,
    decondition,
    fix,
    unfix,
    conditioned,
    OrderedDict,
    ordered,                # Exports from Bijectors
    maximum_a_posteriori,
    maximum_likelihood,
    # The MAP and MLE exports are only needed for the Optim.jl interface.
    MAP,
    MLE

end
