module Inference

using ..Samplers
using ..Core.VarReplay
using Distributions
import ..Turing
using ..Turing: DEFAULT_ADAPT_CONF_TYPE, STAN_DEFAULT_ADAPT_CONF, PROGRESS, CACHERESET
import Distributions: sample
using ..Core.Container
using ..Utilities
import ..Utilities: Sample
using Libtask
using Bijectors
using ..Core.AD
using ProgressMeter
using ..Turing: runmodel!
using LinearAlgebra
using StatsFuns: logsumexp

# Adaptation
include("Adapt/Adapt.jl")
using .Adapt

# Sampling
include("sample.jl")

# Helper functions
include("hmc_core.jl")

# Concrete algorithm implementations.
include("hmcda.jl")
include("nuts.jl")
include("sghmc.jl")
include("sgld.jl")
include("hmc.jl")
if isdefined(Turing, :DynamicHMC)
    include("dynamichmc.jl")
end
include("mh.jl")
include("is.jl")
include("smc.jl")
include("pgibbs.jl")
include("pmmh.jl")
include("ipmcmc.jl")
include("gibbs.jl")

include("fallbacks.jl")

end # module 
