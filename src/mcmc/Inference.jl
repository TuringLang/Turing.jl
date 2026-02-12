module Inference

using DynamicPPL:
    DynamicPPL,
    @model,
    VarInfo,
    LogDensityFunction,
    AbstractVarInfo,
    setindex!!,
    push!!,
    setlogp!!,
    getlogjoint,
    getlogjoint_internal,
    VarName,
    getsym,
    Model,
    DefaultContext
using Distributions, Libtask, Bijectors
using DistributionsAD: VectorOfMultivariate
using LinearAlgebra
using ..Turing: PROGRESS, Turing
using StatsFuns: logsumexp
using Random: AbstractRNG
using AbstractMCMC: AbstractModel, AbstractSampler
using DocStringExtensions: FIELDS, TYPEDEF, TYPEDFIELDS
using DataStructures: OrderedSet, OrderedDict

import ADTypes
import AbstractMCMC
import AbstractPPL
import AdvancedHMC
const AHMC = AdvancedHMC
import AdvancedMH
const AMH = AdvancedMH
import AdvancedPS
import EllipticalSliceSampling
import LogDensityProblems
import Random
import MCMCChains
import StatsBase: predict

export Hamiltonian,
    StaticHamiltonian,
    AdaptiveHamiltonian,
    MH,
    LinkedRW,
    ESS,
    Emcee,
    Gibbs,
    GibbsConditional,
    HMC,
    SGLD,
    PolynomialStepsize,
    SGHMC,
    HMCDA,
    NUTS,
    SMC,
    CSMC,
    PG,
    RepeatSampler,
    Prior,
    externalsampler,
    init_strategy,
    loadstate

const DEFAULT_CHAIN_TYPE = MCMCChains.Chains

include("abstractmcmc.jl")
include("repeat_sampler.jl")
include("external_sampler.jl")

# Directly overload the constructor of `AbstractMCMC.ParamsWithStats` so that we don't
# hit the default method, which uses `getparams(state)` and `getstats(state)`. For Turing's
# MCMC samplers, the state might contain results that are in linked space. Using the
# outputs of the transition here ensures that parameters and logprobs are provided in
# user space (similar to chains output).
function AbstractMCMC.ParamsWithStats(
    model,
    sampler,
    transition::DynamicPPL.ParamsWithStats,
    state;
    params::Bool=true,
    stats::Bool=false,
    extras::Bool=false,
)
    p = params ? [string(k) => v for (k, v) in pairs(transition.params)] : nothing
    s = stats ? transition.stats : NamedTuple()
    e = extras ? NamedTuple() : NamedTuple()
    return AbstractMCMC.ParamsWithStats(p, s, e)
end

#######################################
# Concrete algorithm implementations. #
#######################################

include("ess.jl")
include("hmc.jl")
include("mh.jl")
include("is.jl")
include("particle_mcmc.jl")
include("sghmc.jl")
include("emcee.jl")
include("prior.jl")

include("gibbs.jl")
include("gibbs_conditional.jl")

end # module
