"""
------------------------------------
not working yet
still being built
refering ess.jl which is similar
-------------------------------------

Nested sampling algorithm.

An example
#Note-to-self: set a seed here
julia> using NestedSamplers
julia> using Distributions

# eggbox likelihood function
tmax = 3π
julia> function logl(x)
       t = @. 2*tmax*x - tmax
       return 2 + cos(t[1]/2)*cos(t[2]/2)^5
       end
logl (generic function with 1 method)

# prior constrined to line in the range (0, 20)
julia> prior = [
           Uniform(0, 20),
           Uniform(0, 20)
       ]
2-element Array{Uniform{Float64},1}:
 Uniform{Float64}(a=0.0, b=20.0)
 Uniform{Float64}(a=0.0, b=20.0)

# creating the model
julia> model = NestedModel(logl, prior)
NestedModel{typeof(logl),Uniform{Float64}}(logl, Uniform{Float64}[Uniform{Float64}(a=0.0, b=20.0), Uniform{Float64}(a=0.0, b=20.0)])

julia> using StatsBase: sample, Weights
julia> using MCMCChains: Chains

# creating the sampler
# 2 parameters, 100 active points, multi-ellipsoid

julia> spl = Nested(2, 100; bounds = Bounds.MultiEllipsoid)
Nested(ndims=2, nactive=100, enlarge=1.25, update_interval=150)
  bounds=MultiEllipsoid{Float64}(ndims=2)
  proposal=NestedSamplers.Proposals.Uniform
  logz=-1.0e300
  log_vol=-4.610166019324897
  H=0.0

julia> spl = Nested(2, 100; bounds = Bounds.MultiEllipsoid)
Nested(ndims=2, nactive=100, enlarge=1.25, update_interval=150)
  bounds=MultiEllipsoid{Float64}(ndims=2)
  proposal=NestedSamplers.Proposals.Uniform
  logz=-1.0e300
  log_vol=-4.610166019324897
  H=0.0

julia> chain = sample(model, spl;
                      dlogz = 0.2,
                      param_names = ["x", "y"],
                      chain_type = Chains)
Object of type Chains, with data of type 358×3×1 Array{Float64,3}

Log evidence      = 2.086139406693275
Iterations        = 1:358
Thinning interval = 1
Chains            = 1
Samples per chain = 358
internals         = weights
parameters        = x, y

2-element Array{MCMCChains.ChainDataFrame,1}

Summary Statistics
  parameters     mean     std  naive_se    mcse       ess   r_hat
  ──────────  ───────  ──────  ────────  ──────  ────────  ──────
           x   9.9468  5.7734    0.3051  0.5474  315.7439  1.0017
           y  10.0991  5.5946    0.2957  0.2016  396.0076  0.9995

Quantiles
  parameters    2.5%   25.0%   50.0%    75.0%    97.5%
  ──────────  ──────  ──────  ──────  ───────  ───────
           x  0.4682  5.2563  9.5599  14.9525  19.3789
           y  0.5608  5.4179  9.8601  14.8477  19.2020

"""

## what all to setup here:
## 1. A subtype of AbstractSampler, defined as a mutable struct containing state information or sampler parameters
## 2. A function sample_init! which performs any necessary set-up (default: do not perform any set-up)
## 3. A function step! which returns a transition that represents a single draw from the sampler.
## 4. A function transitions_init which returns a container for the transitions obtained from the sampler (default: return a Vector{T} of length N where T is the type of the transition obtained in the first step and N is the number of requested samples).
## 5. A function transitions_save! which saves transitions to the container (default: save the transition of iteration i at position i in the vector of transitions)
## 6. A function sample_end! which handles any sampler wrap-up (default: do not perform any wrap-up)
## 7. A function bundle_samples which accepts the container of transitions and returns a collection of samples (default: return the vector of transitions)

struct NS{space, P, B} <: InferenceAlgorithm 
    ndims::Int    # number of parameters
    nactive::Int    # number of active points
    proposals::P
    bounds::B   
end

proposal(p::NestedSamplers.Proposals) = p
bound(b::NestedSamplers.Bounds) = b

NS() = NS{()}()
NS(space::Symbol) = NS{(space,)}()

isgibbscomponent(::NS) = true # this states that NS alg is allowed as a Gibbs component

mutable struct NSState{V<:VarInfo} <: AbstractSamplerState
       vi::V
end

function Sampler(alg::NS, model::Model, s::Selector)
       # sanity check
       vi = VarInfo(model)
       info = Dict{Symbol, Any}()
       state = NSState(vi)
       info = Dict{Symbol, Any}()
       return Sampler(alg, info, s, state)
end

function AbstractMCMC.step!(
    rng::AbstractRNG,
    model::Model,
    spl::Sampler{<:NS{space, P, B}},
    ::Integer,
    transition;
    kwargs...
) 
       ## incomplete
       where {space, P, B}
    if spl.selector.rerun # Recompute joint in logp
        model(spl.state.vi)
    end
    
    return Transition(spl)
end

function AbstractMCMC.sample_end!(
    ::AbstractRNG,
    ::Model,
    spl::Sampler{<:NS{space, P, B}},
    N::Integer,
    ts::Vector;
    kwargs...
)
    ## incomplete
end


# tilde operators

       
     
