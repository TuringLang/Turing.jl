mutable struct MHState{V<:VarInfo} <: AbstractSamplerState
    proposal_ratio        ::   Float64
    prior_prob            ::   Float64
    violating_support     ::   Bool
    vi                    ::   V
end

MHState(model::Model) = MHState(0.0, 0.0, false, VarInfo(model))
"""
    MH(n_iters::Int)

Metropolis-Hastings sampler.

Usage:

```julia
MH(100, (:m, (x) -> Normal(x, 0.1)))
```

Example:

```julia
# Define a simple Normal model with unknown mean and variance.
@model gdemo(x) = begin
  s ~ InverseGamma(2,3)
  m ~ Normal(0,sqrt(s))
  x[1] ~ Normal(m, sqrt(s))
  x[2] ~ Normal(m, sqrt(s))
  return s, m
end

chn = sample(gdemo([1.5, 2]), MH(1000))
```
"""
mutable struct MH{space} <: InferenceAlgorithm
    proposals ::  Dict{Symbol,Any}  # Proposals for paramters
end

transition_type(spl::Sampler{<:MH}) = typeof(Transition(spl))
alg_str(::Sampler{<:MH}) = "MH"

function MH(proposals::Dict{Symbol, Any}, space::Tuple)
    return MH{space}(proposals)
end

function MH(space...)
    new_space = ()
    proposals = Dict{Symbol,Any}()

    # parse random variables with their hypothetical proposal
    for element in space
        if isa(element, Symbol)
            new_space = (new_space..., element)
        else
            @assert isa(element[1], Symbol) "[MH] ($element[1]) should be a Symbol. For proposal, use the syntax MH(N, (:m, (x) -> Normal(x, 0.1)))"
            new_space = (new_space..., element[1])
            proposals[element[1]] = element[2]
        end
    end
    return MH(proposals, new_space)
end

function Sampler(alg::MH, model::Model, s::Selector)
    alg_str = "MH"

    # Sanity check for space
    if (s.tag == :default) && !isempty(getspace(alg))
        @assert issubset(get_pvars(model), getspace(alg)) "[$alg_str] symbols specified to samplers ($getspace(alg)) doesn't cover the model parameters ($(get_pvars(model)))"
        if !(issetequal(get_pvars(model), getspace(alg)))
            @warn("[$alg_str] extra parameters specified by samplers don't exist in model: $(setdiff(getspace(alg), get_pvars(model)))")
        end
    end

    info = Dict{Symbol, Any}()
    state = MHState(model)

    return Sampler(alg, info, s, state)
end

function propose(model, spl::Sampler{<:MH}, vi::VarInfo)
    spl.state.proposal_ratio = 0.0
    spl.state.prior_prob = 0.0
    spl.state.violating_support = false
    return runmodel!(model, spl.state.vi, spl)
end

# First step always returns a value.
function step!(
    ::AbstractRNG,
    model::Model,
    spl::Sampler{<:MH},
    ::Integer;
    kwargs...
)
    return Transition(spl)
end

# Every step after the first.
function step!(
    ::AbstractRNG,
    model::Model,
    spl::Sampler{<:MH},
    ::Integer,
    ::Transition;
    kwargs...
)
    if spl.selector.tag != :default # Recompute joint in logp
        runmodel!(model, spl.state.vi)
    end
    old_θ = copy(spl.state.vi[spl])
    old_logp = getlogp(spl.state.vi)

    Turing.DEBUG && @debug "Propose new parameters from proposals..."
    propose(model, spl, spl.state.vi)

    Turing.DEBUG && @debug "computing accept rate α..."
    is_accept, _ = mh_accept(-old_logp, -getlogp(spl.state.vi), spl.state.proposal_ratio)

    Turing.DEBUG && @debug "decide wether to accept..."
    if is_accept && !spl.state.violating_support  # accepted
        is_accept = true
    else                      # rejected
        is_accept = false
        spl.state.vi[spl] = old_θ         # reset Θ
        setlogp!(spl.state.vi, old_logp)  # reset logp
    end

    return Transition(spl)
end

function assume(spl::Sampler{<:MH}, dist::Distribution, vn::VarName, vi::VarInfo)
    if isempty(getspace(spl.alg)) || vn.sym in getspace(spl.alg)
        if ~haskey(vi, vn) error("[MH] does not handle stochastic existence yet") end
        old_val = vi[vn]

        if vn.sym in keys(spl.alg.proposals) # Custom proposal for this parameter
            proposal = spl.alg.proposals[vn.sym](old_val)
            if typeof(proposal) == Distributions.Normal{Float64} # If Gaussian proposal
                σ = std(proposal)
                lb = support(dist).lb
                ub = support(dist).ub
                stdG = Normal()
                r = rand(TruncatedNormal(proposal.μ, proposal.σ, lb, ub))
                # cf http://fsaad.scripts.mit.edu/randomseed/metropolis-hastings-sampling-with-gaussian-drift-proposal-on-bounded-support/
                spl.state.proposal_ratio += log(cdf(stdG, (ub-old_val)/σ) - cdf(stdG,(lb-old_val)/σ))
                spl.state.proposal_ratio -= log(cdf(stdG, (ub-r)/σ) - cdf(stdG,(lb-r)/σ))
            else # Other than Gaussian proposal
                r = rand(proposal)
                if !(insupport(dist, r)) # check if value lies in support
                    spl.state.violating_support = true
                    r = old_val
                end
                spl.state.proposal_ratio -= logpdf(proposal, r) # accumulate pdf of proposal
                reverse_proposal = spl.alg.proposals[vn.sym](r)
                spl.state.proposal_ratio += logpdf(reverse_proposal, old_val)
            end

        else # Prior as proposal
            r = rand(dist)
            spl.state.proposal_ratio += (logpdf(dist, old_val) - logpdf(dist, r))
        end

        spl.state.prior_prob += logpdf(dist, r) # accumulate prior for PMMH
        vi[vn] = vectorize(dist, r)
        setgid!(vi, spl.selector, vn)
    else
        r = vi[vn]
    end

    # acclogp!(vi, logpdf(dist, r)) # accumulate pdf of prior
    r, logpdf(dist, r)
end

function assume(  spl::Sampler{<:MH},
                  dists::Vector{D},
                  vn::VarName,
                  var::Any,
                  vi::VarInfo
                ) where D<:Distribution
    error("[Turing] MH doesn't support vectorizing assume statement")
end

function observe(spl::Sampler{<:MH}, d::Distribution, value::Any, vi::VarInfo)
    return observe(nothing, d, value, vi)  # accumulate pdf of likelihood
end

function observe( spl::Sampler{<:MH},
                  ds::Vector{D},
                  value::Any,
                  vi::VarInfo
                )  where D<:Distribution
    return observe(nothing, ds, value, vi) # accumulate pdf of likelihood
end
