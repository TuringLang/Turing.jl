module Turing

using Requires

##############
# Dependency #
########################################################################
# NOTE: when using anything from external packages,                    #
#       let's keep the practice of explictly writing Package.something #
#       to indicate that's not implemented inside Turing.jl            #
########################################################################

function __init__()
  @require CmdStan="593b3428-ca2f-500c-ae53-031589ec8ddd" @eval begin
      using CmdStan
      import CmdStan: Adapt, Hmc
  end

  @require DynamicHMC="bbc10e6e-7c05-544b-b16e-64fede858acb" @eval begin
      using DynamicHMC, LogDensityProblems
      using LogDensityProblems: AbstractLogDensityProblem, ValueGradient

      struct FunctionLogDensity{F} <: AbstractLogDensityProblem
          dimension::Int
          f::F
      end

      LogDensityProblems.dimension(ℓ::FunctionLogDensity) = ℓ.dimension

      LogDensityProblems.logdensity(::Type{ValueGradient}, ℓ::FunctionLogDensity, x) = ℓ.f(x)::ValueGradient
  end
end

# Below is a trick to remove the dependency of Stan by Requires.jl
# Please see https://github.com/TuringLang/Turing.jl/pull/459 for explanations
@static if isdefined(Turing, :CmdStan)
    const DEFAULT_ADAPT_CONF_TYPE = Union{Nothing, CmdStan.Adapt}
    const STAN_DEFAULT_ADAPT_CONF = CmdStan.Adapt()
else
    const DEFAULT_ADAPT_CONF_TYPE = Nothing
    const STAN_DEFAULT_ADAPT_CONF = nothing
end

const PROGRESS = Ref(true)
function turnprogress(switch::Bool)
    @info("[Turing]: global PROGRESS is set as $switch")
    PROGRESS[] = switch
end

# Constants for caching
const CACHERESET  = 0b00 # 0
const CACHERANGES = 0b01 # 1
const CACHEIDCS   = 0b10 # 2

include("model.jl")

include("Utilities/Utilities.jl")
using .Utilities

include("Samplers/Samplers.jl")
using .Samplers

include("Core/Core.jl")
using .Core.VarReplay
using .Core.Compiler
using .Core.Container
using .Core.AD

include("Inference/Inference.jl")
using .Inference
using .Inference.Adapt

using Reexport
using Libtask
@reexport using Distributions
@reexport using MCMCChain

###########
# Exports #
###########

# Turing essentials - modelling macros and inference algorithms
export  # Modelling
        @model, 
        @VarName,

        # Classic sampling
        MH, 
        Gibbs,

        # Hamiltonian-like sampling
        HMC, 
        SGLD, 
        SGHMC, 
        HMCDA, 
        NUTS,
        DynamicNUTS,

        # Particle-based sampling
        IS, 
        SMC, 
        CSMC, 
        PG, 
        PIMH, 
        PMMH, 
        IPMCMC,

        #Inference
        sample,
        setchunksize,
        resume,           

        # Helper
        auto_tune_chunk_size!,
        setadbackend, 
        setadsafe,

        # Debugging
        turnprogress,
        consume,
        produce,

        # Turing-safe data structures and associated functions
        TArray, 
        tzeros, 
        localcopy, 
        IArray,

        # Distributions
        Flat, 
        FlatPos, 
        BinomialLogit, 
        VecBinomialLogit

end
