module Turing

##############################
# Global variables/constants #
##############################

global const NULL = NaN     # constant for "delete" vals

global CHUNKSIZE = 50       # default chunksize used by AD

global VERBOSITY = 1        # verbosity for dprintln & dwarn
global const FCOMPILER = 1  # verbose printing flag for compiler

# Constans for caching
global const CACHERESET  = 0b00
global const CACHEIDCS   = 0b10
global const CACHERANGES = 0b01

##############
# Dependency #
##############

using StatsFuns
using Distributions
using ForwardDiff: Dual, npartials    # for automatic differentiation

abstract InferenceAlgorithm{P}
type Sampler{T<:InferenceAlgorithm}
  alg   ::  T
  info  ::  Dict{Symbol, Any}         # sampler infomation
end

# TODO: make VarInfo into a seperate module?
include("core/varinfo.jl")  # internal variable container
include("trace/trace.jl")   # running probabilistic programs as tasks

using Turing.Traces
using ProgressMeter
@suppress_err begin using Mamba end

import Distributions: sample          # to orverload sample()
import Base: ~, convert, promote_rule

###########
# Exports #
###########

# Turing essentials - modelling macros and inference algorithms
export @model, @~                           # modelling
export HMC, HMCDA, NUTS, IS, SMC, PG, Gibbs # sampling algorithms
export sample, setchunksize                 # inference
export dprintln, set_verbosity              # debugging

# TODO: remove the exports below
export VarName, VarInfo, Sampler            # to be used in @model

# Turing-safe data structures and associated functions
export TArray, tzeros, localcopy, IArray

# Export Mamba Chain utility functions, used for checking results
export describe, plot, write, heideldiag, rafterydiag, gelmandiag

##################
# Turing helpers #
##################

set_verbosity(v::Int) = global VERBOSITY = v

doc"""
    dprintln(v, args...)

Debugging print function. The first argument controls the verbosity of message.
"""
dprintln(v::Int, args...) = v < Turing.VERBOSITY ?
                            println("\r[Turing]: ", args..., "\n $(stacktrace()[2])") :
                            nothing
dwarn(v::Int, args...)    = v < Turing.VERBOSITY ?
                            print_with_color(:red, "\r[Turing.WARNING]: ", mapreduce(string,*,args), "\n $(stacktrace()[2])\n") :
                            nothing
derror(v::Int, args...)   = error("\r[Turing.ERROR]: ", mapreduce(string,*,args))

##################
# Inference code #
##################

include("core/util.jl")         # utility functions
include("core/compiler.jl")     # compiler
include("core/container.jl")    # particle container
include("core/io.jl")           # I/O
include("samplers/sampler.jl")  # samplers
include("core/ad.jl")           # Automatic Differentiation

end
