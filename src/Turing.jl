module Turing

##############################
# Global variables/constants #
##############################

global const NULL = NaN     # constant for "delete" vals

global CHUNKSIZE = 50       # default chunksize used by AD
setchunksize(chunk_size::Int) = begin
  println("[Turing]: AD chunk size is set as $chunk_size")
  global CHUNKSIZE = chunk_size
end

global PROGRESS = true
turnprogress(switch::Bool) = begin
  println("[Turing]: global PROGRESS is set as $switch")
  global PROGRESS = switch
end

global VERBOSITY = 1        # verbosity for dprintln & dwarn
global const FCOMPILER = 0  # verbose printing flag for compiler

# Constans for caching
global const CACHERESET  = 0b00
global const CACHEIDCS   = 0b10
global const CACHERANGES = 0b01

##############
# Dependency #
########################################################################
# NOTE: when using anything from external packages,                    #
#       let's keep the practice of explictly writing Package.something #
#       to indicate that's not implemented inside Turing.jl            #
########################################################################

using Distributions
using ForwardDiff
using ProgressMeter

import Base: ~, convert, promote_rule, string, isequal, ==, hash, getindex, setindex!, push!, rand, show, isnan, isempty
import Distributions: sample
import ForwardDiff: gradient
import Mamba: AbstractChains, Chains

abstract InferenceAlgorithm
abstract Hamiltonian <: InferenceAlgorithm

doc"""
    Sampler{T}

Generic interface for implementing inference algorithms.
An implementation of an algorithm should include the following:

1. A type specifying the algorithm and its parameters, derived from InferenceAlgorithm
2. A method of `sample` function that produces results of inference, which is where actual inference happens.

Turing translates models to chunks that call the modelling functions at specified points. The dispatch is based on the value of a `sampler` variable. To include a new inference algorithm implements the requirements mentioned above in a separate file,
then include that file at the end of this one.
"""
type Sampler{T<:InferenceAlgorithm}
  alg   ::  T
  info  ::  Dict{Symbol, Any}         # sampler infomation
end

# TODO: make VarInfo into a seperate module?
include("core/varinfo.jl")  # core internal variable container
include("trace/trace.jl")   # to run probabilistic programs as tasks

using Turing.Traces

###########
# Exports #
###########

# Turing essentials - modelling macros and inference algorithms
export @model, @~, @VarName                  # modelling
export HMC, HMCDA, NUTS, IS, SMC, PG, Gibbs  # sampling algorithms
export sample, setchunksize, resume          # inference
export dprintln, set_verbosity, turnprogress # debugging

# Turing-safe data structures and associated functions
export TArray, tzeros, localcopy, IArray

export @sym_str

export UnivariateGMM2, Flat, FlatPos

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
