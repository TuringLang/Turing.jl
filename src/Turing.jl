module Turing

##############
# Dependency #
########################################################################
# NOTE: when using anything from external packages,                    #
#       let's keep the practice of explictly writing Package.something #
#       to indicate that's not implemented inside Turing.jl            #
########################################################################

using Requires, Reexport, ForwardDiff
using Bijectors, StatsFuns, SpecialFunctions
using Statistics, LinearAlgebra, ProgressMeter
using Markdown, Libtask, MacroTools
using AbstractMCMC
@reexport using Distributions, MCMCChains, Libtask
using Tracker: Tracker

import Base: ~, ==, convert, hash, promote_rule, rand, getindex, setindex!
import MCMCChains: AbstractChains, Chains
import DynamicPPL: getspace, runmodel!

const PROGRESS = Ref(true)
function turnprogress(switch::Bool)
    @info("[Turing]: global PROGRESS is set as $switch")
    PROGRESS[] = switch
end

const DEBUG = Bool(parse(Int, get(ENV, "DEBUG_TURING", "0")))

# Random probability measures.
include("stdlib/distributions.jl")
include("stdlib/RandomMeasures.jl")
include("utilities/Utilities.jl")
using .Utilities
include("core/Core.jl")
using .Core
include("inference/Inference.jl")  # inference algorithms
using .Inference
include("variational/VariationalInference.jl")
using .Variational

# TODO: re-design `sample` interface in MCMCChains, which unify CmdStan and Turing.
#   Related: https://github.com/TuringLang/Turing.jl/issues/746
#@init @require CmdStan="593b3428-ca2f-500c-ae53-031589ec8ddd" @eval begin
#     @eval Utilities begin
#         using ..Turing.CmdStan: CmdStan, Adapt, Hmc
#         using ..Turing: HMC, HMCDA, NUTS
#         include("utilities/stan-interface.jl")
#     end
# end

@init @require DynamicHMC="bbc10e6e-7c05-544b-b16e-64fede858acb" @eval Inference begin
    using Pkg; 
    Pkg.installed()["DynamicHMC"] < v"2.0" && error("Please upgdate your DynamicHMC, v1.x is no longer supported")
    using ..Turing.DynamicHMC: DynamicHMC, mcmc_with_warmup
    include("contrib/inference/dynamichmc.jl")
end

###########
# Exports #
###########

# Turing essentials - modelling macros and inference algorithms
export  @model,                 # modelling
        @varname,
        @varinfo,
        @logpdf,
        @sampler,
        DynamicPPL,

        MH,                     # classic sampling
        ESS,
        Gibbs,

        HMC,                    # Hamiltonian-like sampling
        SGLD,
        SGHMC,
        HMCDA,
        NUTS,
        DynamicNUTS,
        ANUTS,

        IS,                     # particle-based sampling
        SMC,
        CSMC,
        PG,
        PIMH,
        PMMH,
        IPMCMC,

        vi,                    # variational inference
        ADVI,

        sample,                 # inference
        psample,
        setchunksize,
        resume,
        @logprob_str,
        @prob_str,

        auto_tune_chunk_size!,  # helper
        setadbackend,
        setadsafe,

        turnprogress,           # debugging

        Flat,
        FlatPos,
        BinomialLogit,
        VecBinomialLogit,
        OrderedLogistic,
        LogPoisson,
        NamedDist

end
