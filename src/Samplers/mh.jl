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
    n_iters   ::  Int       # number of iterations
    proposals ::  Dict{Symbol,Any}  # Proposals for paramters
    gid       ::  Int       # group ID
end
function MH(n_iters::Int, space...)
    new_space = build_mh_space(space)
    proposals = Dict{Symbol,Any}(get_mh_proposals(space)...)
    MH{new_space}(n_iters, proposals, 0)
end
MH{T}(alg::MH, new_gid::Int) where T = MH{T}(alg.n_iters, alg.proposals, new_gid)

@inline function build_mh_space(s::Tuple)
    if length(s) == 0
        return ()
    elseif s[1] isa Symbol
        return (s[1], build_mh_space(Base.tail(s))...)
    else
        @assert s[1][1] isa Symbol "[MH] ($s[1]) should be a Symbol. For proposal, use the syntax MH(N, (:m, (x) -> Normal(x, 0.1)))"
        return (s[1][1], build_mh_space(Base.tail(s))...)
    end    
end
@inline function get_mh_proposals(s::Tuple)
    if length(s) == 0
        return ()
    elseif !(s[1] isa Symbol) && Base.haslength(s[1])
        @assert s[1][1] isa Symbol "[MH] ($s[1]) should be a Symbol. For proposal, use the syntax MH(N, (:m, (x) -> Normal(x, 0.1)))"
        return ((s[1][1] => s[1][2]), get_mh_proposals(Base.tail(s))...)
    else
        return get_mh_proposals(Base.tail(s))
    end    
end

function Sampler(alg::MH, model::Model)
    alg_str = "MH"
    # Sanity check for space
    if alg.gid == 0 && !isempty(getspace(alg))
        verifyspace(getspace(alg), model.pvars, alg_str)
    end
    info = Dict{Symbol, Any}()
    info[:proposal_ratio] = 0.0
    info[:prior_prob] = 0.0
    info[:violating_support] = false

    Sampler(alg, info)
end
