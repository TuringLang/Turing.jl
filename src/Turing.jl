module Turing
__precompile__(false)

##############
# Dependency #
########################################################################
# NOTE: when using anything from external packages,                    #
#       let's keep the practice of explictly writing Package.something #
#       to indicate that's not implemented inside Turing.jl            #
########################################################################

using Requires
using Reexport
@reexport using Distributions
@reexport using MCMCChain
using ForwardDiff
using StatsFuns

using LinearAlgebra
using ProgressMeter
using Markdown
using Libtask
using MacroTools

#  @init @require Stan="682df890-35be-576f-97d0-3d8c8b33a550" begin
using Stan
import Stan: Adapt, Hmc
#  end
import Base: ~, convert, promote_rule, rand, getindex, setindex!
import Distributions: sample
import ForwardDiff: gradient
using Flux: Tracker
import MCMCChain: AbstractChains, Chains

##############################
# Global variables/constants #
##############################

const ADBACKEND = Ref(:reverse_diff)
setadbackend(backend_sym) = begin
  @assert backend_sym == :forward_diff || backend_sym == :reverse_diff
  backend_sym == :forward_diff && CHUNKSIZE[] == 0 && setchunksize(40)
  ADBACKEND[] = backend_sym
end

const ADSAFE = Ref(false)
setadsafe(switch::Bool) = begin
  @info("[Turing]: global ADSAFE is set as $switch")
  ADSAFE[] = switch
end

const CHUNKSIZE = Ref(40) # default chunksize used by AD

setchunksize(chunk_size::Int) = begin
  if ~(CHUNKSIZE[] == chunk_size)
    @info("[Turing]: AD chunk size is set as $chunk_size")
    CHUNKSIZE[] = chunk_size
  end
end

const PROGRESS = Ref(true)
turnprogress(switch::Bool) = begin
  @info("[Turing]: global PROGRESS is set as $switch")
  PROGRESS[] = switch
end

# Constants for caching
const CACHERESET  = 0b00
const CACHEIDCS   = 0b10
const CACHERANGES = 0b01

#######################
# Sampler abstraction #
#######################

abstract type InferenceAlgorithm end
abstract type Hamiltonian <: InferenceAlgorithm end
abstract type AbstractSampler end
"""
    Sampler{T}

Generic interface for implementing inference algorithms.
An implementation of an algorithm should include the following:

1. A type specifying the algorithm and its parameters, derived from InferenceAlgorithm
2. A method of `sample` function that produces results of inference, which is where actual inference happens.

Turing translates models to chunks that call the modelling functions at specified points. The dispatch is based on the value of a `sampler` variable. To include a new inference algorithm implements the requirements mentioned above in a separate file,
then include that file at the end of this one.
"""
mutable struct Sampler{T<:InferenceAlgorithm} <: AbstractSampler
  alg   ::  T
  info  ::  Dict{Symbol, Any}         # sampler infomation
end

"""
Robust initialization method for model parameters in Hamiltonian samplers.
"""
struct HamiltonianRobustInit <: AbstractSampler end
struct SampleFromPrior <: AbstractSampler end

# This can be removed when all `spl=nothing` is replaced with
#   `spl=SampleFromPrior`
const AnySampler = Union{Nothing, AbstractSampler}

include("utilities/helper.jl")
include("utilities/transform.jl")
include("utilities/util.jl")         # utility functions
include("utilities/io.jl")           # I/O
include("core/varinfo.jl")  # core internal variable container
include("core/trace.jl")   # to run probabilistic programs as tasks

using Turing.VarReplay

###########
# Exports #
###########

# Turing essentials - modelling macros and inference algorithms
export @model, @~, @VarName                   # modelling
export MH, Gibbs                              # classic sampling
export HMC, SGLD, SGHMC, HMCDA, NUTS          # Hamiltonian-like sampling
export IS, SMC, CSMC, PG, PIMH, PMMH, IPMCMC  # particle-based sampling
export sample, setchunksize, resume           # inference
export auto_tune_chunk_size!, setadbackend, setadsafe # helper
export turnprogress  # debugging
export consume, produce

# Turing-safe data structures and associated functions
export TArray, tzeros, localcopy, IArray

export @sym_str

export Flat, FlatPos

##################
# Inference code #
##################

include("core/compiler.jl")     # compiler
include("core/container.jl")    # particle container
include("samplers/sampler.jl")  # samplers
include("core/ad.jl")           # Automatic Differentiation

end
