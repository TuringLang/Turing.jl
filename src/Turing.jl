module Turing

using Distributions
using ForwardDiff: Dual, npartials  # for automatic differentiation

abstract InferenceAlgorithm{P}
abstract Sampler{T<:InferenceAlgorithm}

# Code associated with running probabilistic programs as tasks. REVIEW: can we find a way to move this to where the other included files locate.
include("core/varinfo.jl")
include("trace/trace.jl")
using Turing.Traces

import Distributions: sample        # to orverload sample()
import Base: ~, convert, promote_rule
@suppress_err begin using Mamba end

#################
# Turing module #
#################

# Turing essentials - modelling macros and inference algorithms
export @model, @~, InferenceAlgorithm, HMC, IS, SMC, PG, Gibbs, sample, Chain, Sample, Sampler, ImportanceSampler, HMCSampler, setchunksize
export VarName, VarInfo, nextvn, randr, randoc, retain, groupvals
export Dual

# Export Mamba Chain utility functions
export describe, plot, write, heideldiag, rafterydiag, gelmandiag

# Turing-safe data structures and associated functions
export TArray, tzeros, localcopy, IArray

# Debugging helpers
export dprintln

# Global data structures
const TURING = Dict{Symbol, Any}()
global sampler = nothing
global debug_level = 0
global CHUNKSIZE = 50

##########
# Helper #
##########
doc"""
    dprintln(v, args...)

Debugging print function: The first argument controls the verbosity of message, e.g. larger v leads to more verbose debugging messages.
"""
dprintln(v, args...) = v < Turing.debug_level ? println(args...) : nothing

##################
# Inference code #
##################
include("core/util.jl")
include("core/compiler.jl")
include("core/container.jl")
include("core/io.jl")
include("samplers/sampler.jl")

include("core/ad.jl")

end
