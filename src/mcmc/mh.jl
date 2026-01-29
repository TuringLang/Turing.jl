using AdvancedMH: AdvancedMH

"""
    MH(vn1 => proposal1, vn2 => proposal2, ...)

Construct a Metropolis-Hastings algorithm.

Each argument `proposal` can be

- Blank (i.e. `MH()`), in which case `MH` defaults to using the prior for each parameter as
  the proposal distribution.
- A mapping of `VarName`s to a `Distribution`, `LinkedRW`, or a generic callable that
  defines a conditional proposal distribution.


    MH(cov_matrix)

Construct a Metropolis-Hastings algorithm that performs random-walk sampling in linked
space, with proposals drawn from a multivariate normal distribution with the given
covariance matrix.

# Examples

Consider the model below:

```julia
@model function gdemo()
    s ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s))
    1.5 ~ Normal(m, sqrt(s))
    2.0 ~ Normal(m, sqrt(s))
end
```

The default constructor, `MH()`, uses the prior distributions as proposals. So, new
proposals are obtained by sampling `s` from `InverseGamma(2,3)` and `m` from `Normal(0,
sqrt(s))`.

```julia
spl = MH()
```

Alternatively, a mapping of variable names to proposal distributions can be provided.
This implies the use of static proposals for each variable. If a variable is not specified,
its prior distribution is used as the proposal.

```julia
# Use a static proposal for s² (which happens to be the same as the prior) and a static
# proposal for m (note that this isn't a random walk proposal).
spl = MH(
    # This happens to be the same as the prior
    @varname(s) => InverseGamma(2, 3),
    # This is different from the prior
    @varname(m) => Normal(0, 1),
)
```

If the `VarName` of interest is a single symbol, you can also use a `Symbol` instead.

```julia
spl = MH(
    :s => InverseGamma(2, 3),
    :m => Normal(0, 1),
)
```

You can also use a callable to define a proposal that is conditional on the current values.
The callable must accept a single argument, which is a `DynamicPPL.VarNamedTuple` that holds
all the values of the parameters from the previous step. You can obtain the value of a
specific parameter by indexing into this `VarNamedTuple` using a `VarName` (note that symbol
indexing is not supported). The callable must then return a `Distribution` from which to
draw the proposal.

!!! note
    In general, there is no way for Turing to reliably detect whether a proposal is meant to
    be a callable or not, since callable structs may have any type. Hence, any proposal that
    is *not* a distribution is assumed to be a callable.

```julia
spl = MH(
    # This is a static proposal (same as above).
    @varname(s) => InverseGamma(2, 3),
    # This is a conditional proposal, which proposes m from a normal
    # distribution centred at the current value of m, with a standard
    # deviation of 0.5.
    @varname(m) => (vnt -> Normal(vnt[@varname(m)], 0.5)),
)
```

**Note that when using conditional proposals, the values obtained by indexing into the
`VarNamedTuple` are always in unlinked space.** Sometimes, you may want to define a random-walk
proposal in linked space. For this, you can use `LinkedRW` as a proposal, which takes a
covariance matrix as an argument:

```julia
using LinearAlgebra: Diagonal
spl = MH(
    @varname(s) => InverseGamma(2, 3),
    @varname(m) => LinkedRW(Diagonal([0.25]))
)
```

In the above example, `LinkedRW(Diagonal([0.25]))` defines a random-walk proposal for `m` in
linked space. This is in fact the same as the conditional proposal above, because `m` is
already unconstrained, and so linked space and unlinked space are the same for this
variable. However, `s` is constrained to be positive, and so using a `LinkedRW` proposal for
`s` would be different from using a normal proposal in unlinked space (`LinkedRW` will
ensure that the proposals for `s` always remain positive in unlinked space).

```julia
spl = MH(
    @varname(s) => LinkedRW(Diagonal([0.5])),
    @varname(m) => LinkedRW(Diagonal([0.25])),
)
```

Finally, providing just a single covariance matrix will cause `MH` to perform random-walk
sampling in linked space with proposals drawn from a multivariate normal distribution. All
variables are linked in this case. The provided matrix must be positive semi-definite and
square. This example is therefore equivalent to the previous one:

```julia
# Providing a custom variance-covariance matrix
spl = MH(
    [0.50 0;
     0 0.25]
)
```
"""
struct MH{I,L<:DynamicPPL.AbstractLinkStrategy} <: AbstractSampler
    "A function which takes two arguments: (1) the VarNamedTuple of raw values at the
    previous step, and (2) a VarNamedTuple of linked values for any variables that have
    `LinkedRW` proposals; and returns an AbstractInitStrategy. We don't have access to the
    VNTs until the actual sampling, so we have to use a function here; the strategy itself
    will be constructed anew in each sampling step."
    init_strategy_constructor::I
    "Linked variables, i.e., variables which have a `LinkedRW` proposal."
    link_strategy::L
end
# If no proposals are given, then the initialisation strategy to use is always
# `InitFromPrior`.
MH() = MH(Returns(DynamicPPL.InitFromPrior()), DynamicPPL.UnlinkAll())

"""
    LinkedRW(cov_matrix)

Define a random-walk proposal in linked space with the given covariance matrix. Note that
the covariance matrix must correspond exactly to the size of the variable in linked space.
"""
struct LinkedRW{C}
    "The covariance matrix to use for the random-walk proposal in linked space."
    cov_matrix::C
end

"""
    InitFromProposals(proposals::VarNamedTuple, priors::Dict{VarName,Distribution})

An initialisation strategy that samples variables from user-defined proposal distributions.
If a proposal distribution is not found in `proposals`, then we defer to sampling from the
prior, but we additionally cache the prior distribution in `priors` for later use in
proposal density calculations.
"""
struct InitFromProposals{V<:DynamicPPL.VarNamedTuple} <: DynamicPPL.AbstractInitStrategy
    "A mapping of VarNames to Tuple{Bool,Distribution}s that they should be sampled from. If
    the VarName is not in this VarNamedTuple, then it will be sampled from the prior. The
    Bool indicates whether the proposal is in linked space (true, i.e., the strategy should
    return a `LinkedVectorValue`); or in untransformed space (false, i.e., the strategy
    should return an `UntransformedValue`)."
    proposals::V
    "A cache of the prior distributions for any variables that were not given an explicit
    proposal. This is needed to compute the proposal density during MH steps."
    priors::Dict{VarName,Distribution}
end
function DynamicPPL.init(
    rng::Random.AbstractRNG, vn::VarName, prior::Distribution, strategy::InitFromProposals
)
    if haskey(strategy.proposals, vn)
        # this is the proposal that the user wanted
        is_linkedrw, dist = strategy.proposals[vn]
        if is_linkedrw
            transform = DynamicPPL.from_linked_vec_transform(prior)
            linked_vec = rand(rng, dist)
            sz = hasmethod(size, Tuple{typeof(prior)}) ? size(prior) : ()
            return DynamicPPL.LinkedVectorValue(linked_vec, transform, sz)
        else
            # Static or conditional proposal in untransformed space.
            return DynamicPPL.UntransformedValue(rand(rng, dist))
        end
    else
        # No proposal was specified for this variable, so we sample from the prior. We
        # also need to cache the prior for later use in log-proposal density calculations.
        strategy.priors[vn] = prior
        return DynamicPPL.UntransformedValue(rand(rng, prior))
    end
end

const SymOrVNPair = Pair{<:Union{Symbol,VarName},<:Any}

_to_varname(s::Symbol) = DynamicPPL.VarName{s}()
_to_varname(vn::VarName) = vn
_to_varname(x) = throw(ArgumentError("Expected Symbol or VarName, got $(typeof(x))"))

function MH(pair1::SymOrVNPair, pairs::Vararg{SymOrVNPair})
    vn_proposal_pairs = (pair1, pairs...)
    # It is assumed that `raw_vals` is a VarNamedTuple that has all the variables' values
    # already set. We can obtain this by using `ValuesAsInModelAccumulator`. Furthermore,
    # `linked_vals` is a VarNamedTuple that stores a `MHLinkedVal` for any variables that
    # have `LinkedRW` proposals. That in turn is obtained using `MHLinkedValuesAccumulator`.
    function init_strategy_constructor(raw_vals, linked_vals)
        proposals = DynamicPPL.VarNamedTuple()
        for pair in vn_proposal_pairs
            vn, proposal = pair
            # Convert all keys to VarNames.
            vn = _to_varname(vn)
            proposal_dist = if proposal isa Distribution
                # Static proposal.
                (false, proposal)
            elseif proposal isa LinkedRW
                # The distribution we draw from is an MvNormal, centred at the current
                # linked value, and with the given covariance matrix. We also need to add a
                # flag to signal that this is being sampled in linked space.
                (true, MvNormal(linked_vals[vn], proposal.cov_matrix))
            else
                # It's a callable that takes `vnt` and returns a distribution.
                (false, proposal(raw_vals))
            end
            proposals = DynamicPPL.templated_setindex!!(
                proposals, proposal_dist, vn, raw_vals.data[AbstractPPL.getsym(vn)]
            )
        end
        return InitFromProposals(proposals, Dict{VarName,Distribution}())
    end
    linkedrw_vns = Set{VarName}(
        _to_varname(vn) for (vn, proposal) in vn_proposal_pairs if proposal isa LinkedRW
    )
    link_strategy = if isempty(linkedrw_vns)
        DynamicPPL.UnlinkAll()
    else
        DynamicPPL.LinkSome(linkedrw_vns)
    end
    return MH(init_strategy_constructor, link_strategy)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::MH;
    initial_params::DynamicPPL.AbstractInitStrategy,
    discard_sample=false,
    kwargs...,
)
    # Generate and return initial parameters. We need to use VAIMAcc because that will
    # generate the VNT for us that provides the values (as opposed to `vi.values` which
    # stores `AbstractTransformedValues`).
    #
    # TODO(penelopeysm): This in fact could very well be OnlyAccsVarInfo. Indeed, if you
    # only run MH, OnlyAccsVarInfo already works right now. The problem is that using MH
    # inside Gibbs needs a full VarInfo.
    vi = DynamicPPL.VarInfo()
    vi = DynamicPPL.setacc!!(vi, DynamicPPL.ValuesAsInModelAccumulator(false))
    vi = DynamicPPL.setacc!!(vi, MHLinkedValuesAccumulator(spl.link_strategy))
    _, vi = DynamicPPL.init!!(rng, model, vi, initial_params)
    vi = DynamicPPL.update_link_status!!(vi, spl.link_strategy, model)
    transition =
        discard_sample ? nothing : DynamicPPL.ParamsWithStats(vi, (; accepted=true))
    return transition, vi
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::MH,
    old_vi::DynamicPPL.AbstractVarInfo;
    discard_sample=false,
    kwargs...,
)
    old_lp = DynamicPPL.getlogjoint_internal(old_vi)
    # The initialisation strategy that we use to generate a proposal depends on the
    # state from the previous step. We need to extract the raw values and linked values
    # that were used in the previous step.
    old_raw_values = DynamicPPL.getacc(old_vi, Val(:ValuesAsInModel)).values
    old_linked_values = DynamicPPL.getacc(old_vi, Val(MH_ACC_NAME)).values
    init_strategy_given_old = spl.init_strategy_constructor(
        old_raw_values, old_linked_values
    )

    # Evaluate the model with a new proposal.
    # TODO(penelopeysm): This could also be an OnlyAccsVarInfo. However, if we change it to
    # OAVI, we need to be careful to preserve the link status of the old state. Right now
    # that can't be done without traversing the model again. The answer is to move
    # link_strategy into init!! as well.
    new_vi = DynamicPPL.setacc!!(old_vi, DynamicPPL.ValuesAsInModelAccumulator(false))
    new_vi = DynamicPPL.setacc!!(new_vi, MHLinkedValuesAccumulator(spl.link_strategy))
    _, new_vi = DynamicPPL.init!!(rng, model, new_vi, init_strategy_given_old)
    new_lp = DynamicPPL.getlogjoint_internal(new_vi)

    # We need to reconstruct the initialisation strategy for the 'reverse' transition
    # i.e. from new_vi to old_vi. That allows us to calculate the proposal density
    # ratio.
    new_raw_values = DynamicPPL.getacc(new_vi, Val(:ValuesAsInModel)).values
    new_linked_values = DynamicPPL.getacc(new_vi, Val(MH_ACC_NAME)).values
    # For any variables that were not given an explicit proposal, we need to know
    # their priors, which will also go into the proposal density.
    unspecified_priors = if init_strategy_given_old isa InitFromProposals
        init_strategy_given_old.priors
    else
        Dict{VarName,Distribution}()
    end
    init_strategy_given_new = spl.init_strategy_constructor(
        new_raw_values, new_linked_values
    )

    # Calculate the log-acceptance probability.
    log_a = (
        new_lp - old_lp +
        log_proposal_density(old_vi, init_strategy_given_new, unspecified_priors) -
        log_proposal_density(new_vi, init_strategy_given_old, unspecified_priors)
    )

    # Decide whether to accept.
    accepted, vi = if -Random.randexp(rng) < log_a
        true, new_vi
    else
        false, old_vi
    end
    transition =
        discard_sample ? nothing : DynamicPPL.ParamsWithStats(vi, (; accepted=accepted))
    return transition, vi
end

"""
    log_proposal_density(
        old_vi::DynamicPPL.AbstractVarInfo,
        init_strategy_given_new::DynamicPPL.AbstractInitStrategy,
        unspecified_priors::Dict{VarName,Distribution}
    )

Calculate the ratio `g(x|x')` where `g` is the proposal distribution used to generate
`x` (represented by `old_vi`), given the new state `x'`.

If the arguments are switched (i.e., `new_vi` is passed as the first argument, and
`init_strategy_given_old` as the second), the function calculates `g(x'|x)`.
"""
function log_proposal_density(
    vi::DynamicPPL.AbstractVarInfo, ::DynamicPPL.InitFromPrior, ::Dict{VarName,Distribution}
)
    # All samples were drawn from the prior -- in this case g(x|x') = g(x) = prior
    # probability of x.
    return DynamicPPL.getlogprior(vi)
end
function log_proposal_density(
    vi::DynamicPPL.AbstractVarInfo,
    strategy::InitFromProposals,
    unspecified_priors::Dict{VarName,Distribution},
)
    # In this case, the proposal distribution might indeed be conditional, so we need to
    # 'run' the initialisation strategies both ways. Luckily, we don't need to run the model
    # itself, since all the information we need is in the proposals. That is the reason why
    # we have to cache the priors in the InitFromProposals struct -- if any variables were
    # not given an explicit proposal (in `strategy.proposals`) we need to know what their
    # prior was.
    vals = DynamicPPL.getacc(vi, Val(:ValuesAsInModel)).values
    g = 0.0
    for (vn, (is_linkedrw, proposal)) in pairs(strategy.proposals)
        if is_linkedrw
            # LinkedRW proposals end up here, but they are symmetric proposals, so we can
            # skip their contribution.
            continue
        else
            # proposal isa Distribution
            g += logpdf(proposal, vals[vn])
        end
    end
    for (vn, prior) in unspecified_priors
        g += logpdf(prior, vals[vn])
    end
    return g
end

# Accumulator to store linked values; but only the ones that have a LinkedRW proposal.
const MH_ACC_NAME = :MHLinkedValues
struct StoreLinkedValues{L<:DynamicPPL.AbstractLinkStrategy}
    link_strategy::L
end
function (s::StoreLinkedValues)(val, tval, logjac, vn, dist)
    return if DynamicPPL.generate_linked_value(s.link_strategy, vn, tval)
        if tval isa DynamicPPL.LinkedVectorValue
            DynamicPPL.get_internal_value(tval)
        else
            # Need to convert untransformed value to linked value.
            DynamicPPL.to_linked_vec_transform(dist)(val)
        end
    else
        DynamicPPL.DoNotAccumulate()
    end
end
function MHLinkedValuesAccumulator(link_strategy::DynamicPPL.AbstractLinkStrategy)
    return DynamicPPL.VNTAccumulator{MH_ACC_NAME}(StoreLinkedValues(link_strategy))
end

# RWMH can be delegated to AdvancedMH. The type bound is intentionally lax because we just
# let the MvNormal constructor handle it.
function MH(cov_matrix::Any)
    return externalsampler(AdvancedMH.RWMH(MvNormal(cov_matrix)); unconstrained=true)
end
