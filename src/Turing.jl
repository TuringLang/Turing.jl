module Turing

using StatsFuns
using Distributions
using ForwardDiff: Dual, npartials  # for automatic differentiation
using ProgressMeter

abstract InferenceAlgorithm{P}
type Sampler{T<:InferenceAlgorithm}
  alg   ::  T
  info  ::  Dict{Symbol, Any}         # sampler infomation
end

# Code associated with running probabilistic programs as tasks.
#  REVIEW: can we find a way to move this to where the other included files locate.
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
export @model, @~, InferenceAlgorithm, HMC, HMCDA, eNUTS, NUTS, IS, SMC, PG, Gibbs, sample, Chain, Sample, Sampler, setchunksize
export VarName, VarInfo, randr, randoc, retain, groupvals
export Dual

# Export Mamba Chain utility functions
export describe, plot, write, heideldiag, rafterydiag, gelmandiag

# Turing-safe data structures and associated functions
export TArray, tzeros, localcopy, IArray

# Debugging helpers
export dprintln, set_verbosity

# Global data structures
global CHUNKSIZE = 50
global VERBOSITY = 1
set_verbosity(v::Int) = global VERBOSITY = v

##########
# Helper #
##########
doc"""
    dprintln(v, args...)

Debugging print function: The first argument controls the verbosity of message, e.g. larger v leads to more verbose debugging messages.
"""
dprintln(v, args...) = v < Turing.VERBOSITY ? println("[Turing:$(stacktrace()[1])]", args...) : nothing

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
