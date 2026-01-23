module Inference

using DynamicPPL:
    DynamicPPL,
    @model,
    Metadata,
    VarInfo,
    SimpleVarInfo,
    LogDensityFunction,
    AbstractVarInfo,
    # TODO(mhauru) all_varnames_grouped_by_symbol isn't exported by DPPL, because it is only
    # implemented for NTVarInfo. It is used by mh.jl. Either refactor mh.jl to not use it
    # or implement it for other VarInfo types and export it from DPPL.
    all_varnames_grouped_by_symbol,
    syms,
    setindex!!,
    push!!,
    setlogp!!,
    getlogjoint,
    getlogjoint_internal,
    VarName,
    getsym,
    getdist,
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
using Accessors: Accessors

import ADTypes
import AbstractMCMC
import AbstractPPL
import AdvancedHMC
const AHMC = AdvancedHMC
import AdvancedMH
const AMH = AdvancedMH
import AdvancedPS
import Accessors
import EllipticalSliceSampling
import LogDensityProblems
import Random
import MCMCChains
import StatsBase: predict

export Hamiltonian,
    StaticHamiltonian,
    AdaptiveHamiltonian,
    MH,
    ESS,
    Emcee,
    Gibbs,      # classic sampling
    GibbsConditional,  # conditional sampling
    HMC,
    SGLD,
    PolynomialStepsize,
    SGHMC,
    HMCDA,
    NUTS,       # Hamiltonian-like sampling
    IS,
    SMC,
    CSMC,
    PG,
    RepeatSampler,
    Prior,
    predict,
    externalsampler,
    init_strategy,
    loadstate

#########################################
# Generic AbstractMCMC methods dispatch #
#########################################

const DEFAULT_CHAIN_TYPE = MCMCChains.Chains
include("abstractmcmc.jl")

####################
# Sampler wrappers #
####################

include("repeat_sampler.jl")
include("external_sampler.jl")

# TODO: make a nicer `set_namedtuple!` and move these functions to DynamicPPL.
function DynamicPPL.unflatten(vi::DynamicPPL.NTVarInfo, θ::NamedTuple)
    set_namedtuple!(deepcopy(vi), θ)
    return vi
end
function DynamicPPL.unflatten(vi::SimpleVarInfo, θ::NamedTuple)
    return SimpleVarInfo(θ, vi.logp, vi.transformation)
end

"""
    mh_accept(logp_current::Real, logp_proposal::Real, log_proposal_ratio::Real)

Decide if a proposal ``x'`` with log probability ``\\log p(x') = logp_proposal`` and
log proposal ratio ``\\log k(x', x) - \\log k(x, x') = log_proposal_ratio`` in a
Metropolis-Hastings algorithm with Markov kernel ``k(x_t, x_{t+1})`` and current state
``x`` with log probability ``\\log p(x) = logp_current`` is accepted by evaluating the
Metropolis-Hastings acceptance criterion
```math
\\log U \\leq \\log p(x') - \\log p(x) + \\log k(x', x) - \\log k(x, x')
```
for a uniform random number ``U \\in [0, 1)``.
"""
function mh_accept(logp_current::Real, logp_proposal::Real, log_proposal_ratio::Real)
    # replacing log(rand()) with -randexp() yields test errors
    return log(rand()) + logp_current ≤ logp_proposal + log_proposal_ratio
end

# Helper functions for AbstractMCMC callbacks
# Helper to get log probability from VarInfo
function _get_lp(vi::DynamicPPL.AbstractVarInfo)
    lp = DynamicPPL.getlogp(vi)
    return sum(values(lp))
end

# Helper to extract raw parameter values from VarInfo as Vector{<:Real}
function _get_params_vector(vi::DynamicPPL.AbstractVarInfo)
    return vi[:]
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
