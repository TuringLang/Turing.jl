###
### Sampler states
###

proposal(p::AdvancedMH.Proposal) = p
proposal(f::Function) = AdvancedMH.StaticProposal(f)
proposal(d::Distribution) = AdvancedMH.StaticProposal(d)
proposal(cov::AbstractMatrix) = AdvancedMH.RandomWalkProposal(MvNormal(cov))
proposal(x) = error("proposals of type ", typeof(x), " are not supported")

"""
    MH(proposals...)

Construct a Metropolis-Hastings algorithm.

The arguments `proposals` can be

- Blank (i.e. `MH()`), in which case `MH` defaults to using the prior for each parameter as the proposal distribution.
- An iterable of pairs or tuples mapping a `Symbol` to a `AdvancedMH.Proposal`, `Distribution`, or `Function`
  that returns a conditional proposal distribution.
- A covariance matrix to use as for mean-zero multivariate normal proposals.

# Examples

The default `MH` will draw proposal samples from the prior distribution using `AdvancedMH.StaticProposal`.

```julia
@model function gdemo(x, y)
    s² ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    y ~ Normal(m, sqrt(s²))
end

chain = sample(gdemo(1.5, 2.0), MH(), 1_000)
mean(chain)
```

Specifying a single distribution implies the use of static MH:

```julia
# Use a static proposal for s² (which happens to be the same
# as the prior) and a static proposal for m (note that this
# isn't a random walk proposal).
chain = sample(
    gdemo(1.5, 2.0),
    MH(
        :s² => InverseGamma(2, 3),
        :m => Normal(0, 1)
    ),
    1_000
)
mean(chain)
```

Specifying explicit proposals using the `AdvancedMH` interface:

```julia
# Use a static proposal for s² and random walk with proposal
# standard deviation of 0.25 for m.
chain = sample(
    gdemo(1.5, 2.0),
    MH(
        :s² => AdvancedMH.StaticProposal(InverseGamma(2,3)),
        :m => AdvancedMH.RandomWalkProposal(Normal(0, 0.25))
    ),
    1_000
)
mean(chain)
```

Using a custom function to specify a conditional distribution:

```julia
# Use a static proposal for s and and a conditional proposal for m,
# where the proposal is centered around the current sample.
chain = sample(
    gdemo(1.5, 2.0),
    MH(
        :s² => InverseGamma(2, 3),
        :m => x -> Normal(x, 1)
    ),
    1_000
)
mean(chain)
```

Providing a covariance matrix will cause `MH` to perform random-walk
sampling in the transformed space with proposals drawn from a multivariate
normal distribution. The provided matrix must be positive semi-definite and
square:

```julia
# Providing a custom variance-covariance matrix
chain = sample(
    gdemo(1.5, 2.0),
    MH(
        [0.25 0.05;
         0.05 0.50]
    ),
    1_000
)
mean(chain)
```

"""
struct MH{P} <: AbstractSampler
    proposals::P

    function MH(proposals...)
        prop_syms = Symbol[]
        props = AMH.Proposal[]

        for s in proposals
            if s isa Pair || s isa Tuple
                # Check to see whether it's a pair that specifies a kernel
                # or a specific proposal distribution.
                push!(prop_syms, s[1])
                push!(props, proposal(s[2]))
            elseif length(proposals) == 1
                # If we hit this block, check to see if it's
                # a run-of-the-mill proposal or covariance
                # matrix.
                prop = proposal(s)

                # Return early, we got a covariance matrix.
                return new{typeof(prop)}(prop)
            else
                # Try to convert it to a proposal anyways,
                # throw an error if not acceptable.
                prop = proposal(s)
                push!(props, prop)
            end
        end

        proposals = NamedTuple{tuple(prop_syms...)}(tuple(props...))

        return new{typeof(proposals)}(proposals)
    end
end

# Some of the proposals require working in unconstrained space.
transform_maybe(proposal::AMH.Proposal) = proposal
function transform_maybe(proposal::AMH.RandomWalkProposal)
    return AMH.RandomWalkProposal(Bijectors.transformed(proposal.proposal))
end

function MH(model::Model; proposal_type=AMH.StaticProposal)
    priors = DynamicPPL.extract_priors(model)
    props = Tuple([proposal_type(prop) for prop in values(priors)])
    vars = Tuple(map(Symbol, collect(keys(priors))))
    priors = map(transform_maybe, NamedTuple{vars}(props))
    return AMH.MetropolisHastings(priors)
end

"""
    MHState(varinfo::AbstractVarInfo, logjoint_internal::Real)

State for Metropolis-Hastings sampling.

`varinfo` must have the correct parameters set inside it, but its other fields
(e.g. accumulators, which track logp) can in general be missing or incorrect.

`logjoint_internal` is the log joint probability of the model, evaluated using
the parameters and linking status of `varinfo`. It should be equal to
`DynamicPPL.getlogjoint_internal(varinfo)`. This information is returned by the
MH sampler so we store this here to avoid re-evaluating the model
unnecessarily.
"""
struct MHState{V<:AbstractVarInfo,L<:Real}
    varinfo::V
    logjoint_internal::L
end

get_varinfo(s::MHState) = s.varinfo

#####################
# Utility functions #
#####################

"""
    set_namedtuple!(vi::VarInfo, nt::NamedTuple)

Places the values of a `NamedTuple` into the relevant places of a `VarInfo`.
"""
function set_namedtuple!(vi::DynamicPPL.VarInfoOrThreadSafeVarInfo, nt::NamedTuple)
    for (n, vals) in pairs(nt)
        vns = vi.metadata[n].vns
        if vals isa AbstractVector
            vals = unvectorize(vals)
        end
        if length(vns) == 1
            # Only one variable, assign the values to it
            DynamicPPL.setindex!(vi, vals, vns[1])
        else
            # Spread the values across the variables
            length(vns) == length(vals) || error("Unequal number of variables and values")
            for (vn, val) in zip(vns, vals)
                DynamicPPL.setindex!(vi, val, vn)
            end
        end
    end
end

# NOTE(penelopeysm): MH does not conform to the usual LogDensityProblems
# interface in that it gets evaluated with a NamedTuple. Hence we need this
# method just to deal with MH.
function LogDensityProblems.logdensity(f::LogDensityFunction, x::NamedTuple)
    vi = deepcopy(f.varinfo)
    # Note that the NamedTuple `x` does NOT conform to the structure required for
    # `InitFromParams`. In particular, for models that look like this:
    #
    # @model function f()
    #     v = Vector{Vector{Float64}}
    #     v[1] ~ MvNormal(zeros(2), I)
    # end
    #
    # `InitFromParams` will expect Dict(@varname(v[1]) => [x1, x2]), but `x` will have the
    # format `(v = [x1, x2])`. Hence we still need this `set_namedtuple!` function.
    #
    # In general `init!!(f.model, vi, InitFromParams(x))` will work iff the model only
    # contains 'basic' varnames.
    set_namedtuple!(vi, x)
    # Update log probability.
    _, vi_new = DynamicPPL.evaluate!!(f.model, vi)
    lj = f.getlogdensity(vi_new)
    return lj
end

# unpack a vector if possible
unvectorize(dists::AbstractVector) = length(dists) == 1 ? first(dists) : dists

# possibly unpack and reshape samples according to the prior distribution
function reconstruct(dist::Distribution, val::AbstractVector)
    return DynamicPPL.from_vec_transform(dist)(val)
end
reconstruct(dist::AbstractVector{<:UnivariateDistribution}, val::AbstractVector) = val
function reconstruct(dist::AbstractVector{<:MultivariateDistribution}, val::AbstractVector)
    offset = 0
    return map(dist) do d
        n = length(d)
        newoffset = offset + n
        v = val[(offset + 1):newoffset]
        offset = newoffset
        return v
    end
end

"""
    dist_val_tuple(spl::MH, vi::VarInfo)

Return two `NamedTuples`.

The first `NamedTuple` has symbols as keys and distributions as values.
The second `NamedTuple` has model symbols as keys and their stored values as values.
"""
function dist_val_tuple(spl::MH, vi::DynamicPPL.VarInfoOrThreadSafeVarInfo)
    vns = all_varnames_grouped_by_symbol(vi)
    dt = _dist_tuple(spl.proposals, vi, vns)
    vt = _val_tuple(vi, vns)
    return dt, vt
end

@generated function _val_tuple(vi::VarInfo, vns::NamedTuple{names}) where {names}
    isempty(names) && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[
        :(
            $name = reconstruct(
                unvectorize(DynamicPPL.getdist.(Ref(vi), vns.$name)),
                DynamicPPL.getindex_internal(vi, vns.$name),
            )
        ) for name in names
    ]
    return expr
end
_val_tuple(::VarInfo, ::Tuple{}) = ()

@generated function _dist_tuple(
    props::NamedTuple{propnames}, vi::VarInfo, vns::NamedTuple{names}
) where {names,propnames}
    isempty(names) === 0 && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[
        if name in propnames
            # We've been given a custom proposal, use that instead.
            :($name = props.$name)
        else
            # Otherwise, use the default proposal.
            :(
                $name = AMH.StaticProposal(
                    unvectorize(DynamicPPL.getdist.(Ref(vi), vns.$name))
                )
            )
        end for name in names
    ]
    return expr
end
_dist_tuple(::@NamedTuple{}, ::VarInfo, ::Tuple{}) = ()

# Utility functions to link
should_link(varinfo, sampler, proposal) = false
function should_link(varinfo, sampler, proposal::NamedTuple{(),Tuple{}})
    # If it's an empty `NamedTuple`, we're using the priors as proposals
    # in which case we shouldn't link.
    return false
end
function should_link(varinfo, sampler, proposal::AdvancedMH.RandomWalkProposal)
    return true
end
# FIXME: This won't be hit unless `vals` are all the exactly same concrete type of `AdvancedMH.RandomWalkProposal`!
function should_link(
    varinfo, sampler, proposal::NamedTuple{names,vals}
) where {names,vals<:NTuple{<:Any,<:AdvancedMH.RandomWalkProposal}}
    return true
end

function maybe_link!!(varinfo, sampler, proposal, model)
    return if should_link(varinfo, sampler, proposal)
        DynamicPPL.link!!(varinfo, model)
    else
        varinfo
    end
end

# Make a proposal if we don't have a covariance proposal matrix (the default).
function propose!!(rng::AbstractRNG, prev_state::MHState, model::Model, spl::MH, proposal)
    vi = prev_state.varinfo
    # Retrieve distribution and value NamedTuples.
    dt, vt = dist_val_tuple(spl, vi)

    # Create a sampler and the previous transition.
    mh_sampler = AMH.MetropolisHastings(dt)
    prev_trans = AMH.Transition(vt, prev_state.logjoint_internal, false)

    # Make a new transition.
    model = DynamicPPL.setleafcontext(model, MHContext(rng))
    densitymodel = AMH.DensityModel(
        Base.Fix1(
            LogDensityProblems.logdensity,
            DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi),
        ),
    )
    trans, _ = AbstractMCMC.step(rng, densitymodel, mh_sampler, prev_trans)
    # trans.params isa NamedTuple
    set_namedtuple!(vi, trans.params)
    # Here, `trans.lp` is equal to `getlogjoint_internal(vi)`. We don't know
    # how to set this back inside vi (without re-evaluating). However, the next
    # MH step will require this information to calculate the acceptance
    # probability, so we return it together with vi.
    return MHState(vi, trans.lp)
end

# Make a proposal if we DO have a covariance proposal matrix.
function propose!!(
    rng::AbstractRNG,
    prev_state::MHState,
    model::Model,
    spl::MH,
    proposal::AdvancedMH.RandomWalkProposal,
)
    vi = prev_state.varinfo
    # If this is the case, we can just draw directly from the proposal
    # matrix.
    vals = vi[:]

    # Create a sampler and the previous transition.
    mh_sampler = AMH.MetropolisHastings(spl.proposals)
    prev_trans = AMH.Transition(vals, prev_state.logjoint_internal, false)

    # Make a new transition.
    model = DynamicPPL.setleafcontext(model, MHContext(rng))
    densitymodel = AMH.DensityModel(
        Base.Fix1(
            LogDensityProblems.logdensity,
            DynamicPPL.LogDensityFunction(model, DynamicPPL.getlogjoint_internal, vi),
        ),
    )
    trans, _ = AbstractMCMC.step(rng, densitymodel, mh_sampler, prev_trans)
    # trans.params isa AbstractVector
    vi = DynamicPPL.unflatten(vi, trans.params)
    # Here, `trans.lp` is equal to `getlogjoint_internal(vi)`. We don't know
    # how to set this back inside vi (without re-evaluating). However, the next
    # MH step will require this information to calculate the acceptance
    # probability, so we return it together with vi.
    return MHState(vi, trans.lp)
end

function Turing.Inference.initialstep(
    rng::AbstractRNG, model::DynamicPPL.Model, spl::MH, vi::AbstractVarInfo; kwargs...
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vi = maybe_link!!(vi, spl, spl.proposals, model)

    return Transition(model, vi, nothing), MHState(vi, DynamicPPL.getlogjoint_internal(vi))
end

function AbstractMCMC.step(
    rng::AbstractRNG, model::DynamicPPL.Model, spl::MH, state::MHState; kwargs...
)
    # Cases:
    # 1. A covariance proposal matrix
    # 2. A bunch of NamedTuples that specify the proposal space
    new_state = propose!!(rng, state, model, spl, spl.proposals)

    return Transition(model, new_state.varinfo, nothing), new_state
end

struct MHContext{R<:AbstractRNG} <: DynamicPPL.AbstractContext
    rng::R
end

function DynamicPPL.tilde_assume!!(
    context::MHContext, right::Distribution, vn::VarName, vi::AbstractVarInfo
)
    # Allow MH to sample new variables from the prior if it's not already present in the
    # VarInfo.
    dispatch_ctx = if haskey(vi, vn)
        DynamicPPL.DefaultContext()
    else
        DynamicPPL.InitContext(context.rng, DynamicPPL.InitFromPrior())
    end
    return DynamicPPL.tilde_assume!!(dispatch_ctx, right, vn, vi)
end
function DynamicPPL.tilde_observe!!(
    ::MHContext, right::Distribution, left, vn::Union{VarName,Nothing}, vi::AbstractVarInfo
)
    return DynamicPPL.tilde_observe!!(DefaultContext(), right, left, vn, vi)
end
