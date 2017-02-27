module Turing

# Code associated with running probabilistic programs as tasks. REVIEW: can we find a way to move this to where the other included files locate.
include("trace/trace.jl")

import Distributions: sample        # to orverload sample()
import Base.~                       # to orverload @~
using ForwardDiff: Dual, npartials  # for automatic differentiation
using Turing.Traces

#################
# Turing module #
#################

# Turing essentials - modelling macros and inference algorithms
export @model, @predict, @~, @isdefined, InferenceAlgorithm, HMC, IS, SMC, PG, Gibbs, sample, Chain, Sample, Sampler, ImportanceSampler, HMCSampler, VarInfo, @predictall

# Turing-safe data structures and associated functions
export TArray, tzeros, localcopy, IArray

# Debugging helpers
export dprintln

# Global data structures
const TURING = Dict{Symbol, Any}()
global sampler = nothing
global debug_level = 0
global CHUNKSIZE = 1000

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
include("core/varinfo.jl")
include("samplers/sampler.jl")

include("core/ad.jl")

end
