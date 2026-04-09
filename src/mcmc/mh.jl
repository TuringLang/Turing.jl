using AdvancedMH: AdvancedMH
using AbstractPPL: @varname

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
# Use a static proposal for s (which happens to be the same as the prior) and a static
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
`VarNamedTuple` are always in untransformed space, which are constrained to the support of
the distribution.** Sometimes, you may want to define a random-walk proposal in
unconstrained (i.e. 'linked') space. For this, you can use `LinkedRW` as a proposal, which
takes a covariance matrix as an argument:

```julia
using LinearAlgebra: Diagonal
spl = MH(
    @varname(s) => InverseGamma(2, 3),
    @varname(m) => LinkedRW(Diagonal([0.25]))
)
```

In the above example, `LinkedRW(Diagonal([0.25]))` defines a random-walk proposal for `m` in
linked space. This is in fact the same as the conditional proposal above, because `m` is
already unconstrained, and so the unconstraining transformation is the identity.

However, `s` is constrained to be positive, and so using a `LinkedRW` proposal for `s` would
be different from using a normal proposal in untransformed space (`LinkedRW` will ensure
that the proposals for `s` always remain positive in untransformed space).

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
struct MH{I,L<:DynamicPPL.AbstractTransformStrategy} <: AbstractSampler
    "A function which takes two arguments: (1) the VarNamedTuple of raw values at the
    previous step, and (2) a VarNamedTuple of linked values for any variables that have
    `LinkedRW` proposals; and returns an AbstractInitStrategy. We don't have access to the
    VNTs until the actual sampling, so we have to use a function here; the strategy itself
    will be constructed anew in each sampling step."
    init_strategy_constructor::I
    "Linked variables, i.e., variables which have a `LinkedRW` proposal."
    transform_strategy::L
    "All variables with a proposal"
    vns_with_proposal::Set{VarName}
end
# If no proposals are given, then the initialisation strategy to use is always
# `InitFromPrior`.
MH() = MH(Returns(DynamicPPL.InitFromPrior()), DynamicPPL.UnlinkAll(), Set{VarName}())

"""
    LinkedRW(cov_matrix)

Define a random-walk proposal in linked space with the given covariance matrix. Note that
the size of the covariance matrix must correspond exactly to the size of the variable in
linked space.

    LinkedRW(variance::Real)

If a `Real` variance is provided, `LinkedRW` will just generate a covariance matrix of
`variance * LinearAlgebra.I`.
"""
struct LinkedRW{C}
    "The covariance matrix to use for the random-walk proposal in linked space."
    cov_matrix::C
end
LinkedRW(var::Real) = LinkedRW(var * I)

"""
    InitFromProposals(proposals::VarNamedTuple, verbose::Bool)

An initialisation strategy that samples variables from user-defined proposal distributions.
If a proposal distribution is not found in `proposals`, then we defer to sampling from the
prior.
"""
struct InitFromProposals{V<:DynamicPPL.VarNamedTuple} <: DynamicPPL.AbstractInitStrategy
    "A mapping of VarNames to Tuple{Bool,Distribution}s that they should be sampled from. If
    the VarName is not in this VarNamedTuple, then it will be sampled from the prior. The
    Bool indicates whether the proposal is in linked space (true, i.e., the strategy should
    return a linked vector value); or in untransformed space (false, i.e., the strategy
    should return an untransformed value)."
    proposals::V
    "Whether to print the proposals as they are being sampled"
    verbose::Bool
end
function DynamicPPL.init(
    rng::Random.AbstractRNG, vn::VarName, prior::Distribution, strategy::InitFromProposals
)
    if haskey(strategy.proposals, vn)
        # this is the proposal that the user wanted
        is_linkedrw, dist = strategy.proposals[vn]
        if strategy.verbose
            if is_linkedrw
                @info "varname $vn: proposal is a LinkedRW with covariance matrix $(dist.Σ)"
            else
                @info "varname $vn: proposal $(strategy.proposals[vn][2])"
            end
        end
        if is_linkedrw
            linked_vec = rand(rng, dist)
            return DynamicPPL.TransformedValue(linked_vec, DynamicPPL.DynamicLink())
        else
            # Static or conditional proposal in untransformed space.
            return DynamicPPL.TransformedValue(rand(rng, dist), DynamicPPL.NoTransform())
        end
    else
        strategy.verbose && @info "varname $vn: no proposal specified, drawing from prior"
        # No proposal was specified for this variable, so we sample from the prior.
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
    # already set. We can obtain this by using `RawValueAccumulator`. Furthermore,
    # `linked_vals` is a VarNamedTuple that stores a `MHLinkedVal` for any variables that
    # have `LinkedRW` proposals. That in turn is obtained using `MHLinkedValuesAccumulator`.
    function init_strategy_constructor(raw_vals, linked_vals)
        proposals = DynamicPPL.VarNamedTuple()
        for pair in vn_proposal_pairs
            # Convert all keys to VarNames.
            vn, proposal = pair
            vn = _to_varname(vn)
            if !haskey(raw_vals, vn)
                continue
            end
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
        return InitFromProposals(proposals, false)
    end
    all_vns = Set{VarName}(_to_varname(pair[1]) for pair in vn_proposal_pairs)
    linkedrw_vns = Set{VarName}(
        _to_varname(vn) for (vn, proposal) in vn_proposal_pairs if proposal isa LinkedRW
    )
    link_strategy = if isempty(linkedrw_vns)
        DynamicPPL.UnlinkAll()
    else
        DynamicPPL.LinkSome(linkedrw_vns, DynamicPPL.UnlinkAll())
    end
    return MH(init_strategy_constructor, link_strategy, all_vns)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::MH;
    initial_params::DynamicPPL.AbstractInitStrategy,
    discard_sample=false,
    verbose=true,
    kwargs...,
)
    # Generate and return initial parameters. We need to use VAIMAcc because that will
    # generate the VNT for us that provides the values (as opposed to `vi.values` which
    # stores `AbstractTransformedValues`).
    vi = DynamicPPL.OnlyAccsVarInfo()
    vi = DynamicPPL.setacc!!(vi, DynamicPPL.RawValueAccumulator(false))
    vi = DynamicPPL.setacc!!(vi, MHLinkedValuesAccumulator())
    vi = DynamicPPL.setacc!!(vi, MHUnspecifiedPriorsAccumulator(spl.vns_with_proposal))
    _, vi = DynamicPPL.init!!(rng, model, vi, initial_params, spl.transform_strategy)

    # Since our initial parameters are sampled with `initial_params`, which could be
    # anything, it's possible that the initial parameters are outside the support of the
    # proposal. That will mess up the sampling because when calculating the proposal density
    # ratio, we will get -Inf for the forward proposal density (i.e., log(g(x|x'))), because
    # `log(g(x))` is already -Inf regardless of what `x'` is. We insert a check for this
    # here.
    initial_raw_values = DynamicPPL.get_raw_values(vi)
    initial_linked_values = DynamicPPL.getacc(vi, Val(MH_ACC_NAME)).values
    init_strategy = spl.init_strategy_constructor(initial_raw_values, initial_linked_values)
    initial_unspecified_priors = DynamicPPL.getacc(vi, Val(MH_PRIOR_ACC_NAME)).values
    initial_log_proposal_density = log_proposal_density(
        vi, init_strategy, initial_unspecified_priors
    )
    if initial_log_proposal_density == -Inf || isnan(initial_log_proposal_density)
        io = IOContext(IOBuffer(), :color => true)
        show(io, "text/plain", initial_raw_values)
        init_str = String(take!(io.io))
        prob_dens_string = if initial_log_proposal_density == -Inf
            "zero"
        else
            "a NaN"
        end
        error(
            "The initial parameters have $prob_dens_string probability density under" *
            " the proposal distribution (for example, an initial value of `x=2.0`" *
            " for a proposal `@varname(x) => Uniform(0, 1)`. This will cause the" *
            " sampler to get stuck at the initial parameters. Consider specifying" *
            " different initial parameters (e.g. via `InitFromParams`) or using a" *
            " different proposal distribution." *
            " Your initial values were:\n\n$init_str\n",
        )
    end

    # We evaluate the model once with the sampler's init strategy and print all the
    # proposals that were used. This helps the user detect cases where the proposals are
    # silently ignored (e.g. because the VarName in the proposal doesn't match the VarName
    # in the model).
    if verbose && init_strategy isa InitFromProposals
        @info "When sampling with MH, the following proposals will be used at each step.\nThis output can be disabled by passing `verbose=false` to `sample()`."
        verbose_init_strategy = InitFromProposals(init_strategy.proposals, true)
        oavi = DynamicPPL.OnlyAccsVarInfo(()) # No need to accumulate anything
        DynamicPPL.init!!(rng, model, oavi, verbose_init_strategy, DynamicPPL.UnlinkAll())
    end

    transition =
        discard_sample ? nothing : DynamicPPL.ParamsWithStats(vi, (; accepted=true))
    return transition, vi
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    spl::MH,
    old_vi::DynamicPPL.OnlyAccsVarInfo;
    discard_sample=false,
    kwargs...,
)
    old_lp = DynamicPPL.getlogjoint_internal(old_vi)
    # The initialisation strategy that we use to generate a proposal depends on the
    # state from the previous step. We need to extract the raw values and linked values
    # that were used in the previous step.
    old_raw_values = DynamicPPL.get_raw_values(old_vi)
    old_linked_values = DynamicPPL.getacc(old_vi, Val(MH_ACC_NAME)).values
    old_unspecified_priors = DynamicPPL.getacc(old_vi, Val(MH_PRIOR_ACC_NAME)).values

    init_strategy_given_old = spl.init_strategy_constructor(
        old_raw_values, old_linked_values
    )

    # Evaluate the model with a new proposal.
    new_vi = DynamicPPL.OnlyAccsVarInfo()
    new_vi = DynamicPPL.setacc!!(new_vi, DynamicPPL.RawValueAccumulator(false))
    new_vi = DynamicPPL.setacc!!(new_vi, MHLinkedValuesAccumulator())
    new_vi = DynamicPPL.setacc!!(
        new_vi, MHUnspecifiedPriorsAccumulator(spl.vns_with_proposal)
    )
    _, new_vi = DynamicPPL.init!!(
        rng, model, new_vi, init_strategy_given_old, spl.transform_strategy
    )
    new_lp = DynamicPPL.getlogjoint_internal(new_vi)
    # We need to reconstruct the initialisation strategy for the 'reverse' transition
    # i.e. from new_vi to old_vi. That allows us to calculate the proposal density
    # ratio.
    new_raw_values = DynamicPPL.get_raw_values(new_vi)
    new_linked_values = DynamicPPL.getacc(new_vi, Val(MH_ACC_NAME)).values
    new_unspecified_priors = DynamicPPL.getacc(new_vi, Val(MH_PRIOR_ACC_NAME)).values

    init_strategy_given_new = spl.init_strategy_constructor(
        new_raw_values, new_linked_values
    )

    # Calculate the log-acceptance probability.
    log_a = (
        new_lp - old_lp +
        log_proposal_density(old_vi, init_strategy_given_new, old_unspecified_priors) -
        log_proposal_density(new_vi, init_strategy_given_old, new_unspecified_priors)
    )
    if isnan(log_a)
        @warn "MH log-acceptance probability is NaN; sample will be rejected"
        @info new_lp
        @info old_lp
        @info log_proposal_density(old_vi, init_strategy_given_new, old_unspecified_priors)
        @info log_proposal_density(new_vi, init_strategy_given_old, new_unspecified_priors)
    end

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
        old_vi::DynamicPPL.OnlyAccsVarInfo,
        init_strategy_given_new::DynamicPPL.AbstractInitStrategy,
        old_unspecified_priors::DynamicPPL.VarNamedTuple
    )

Calculate the ratio `g(x|x')` where `g` is the proposal distribution used to generate
`x` (represented by `old_vi`), given the new state `x'`.

If the arguments are switched (i.e., `new_vi` is passed as the first argument, and
`init_strategy_given_old` as the second), the function calculates `g(x'|x)`.

The log-density of the proposal distribution is calculated by summing up the contributions
from:

- any variables that have an explicit proposal in `init_strategy_given_new` (i.e., those
  in `spl.vns_with_proposal`), which can be either static or conditional proposals; and
- any variables that do not have an explicit proposal, for which we defer to its prior
  distribution.
"""
function log_proposal_density(
    vi::DynamicPPL.OnlyAccsVarInfo, ::DynamicPPL.InitFromPrior, ::DynamicPPL.VarNamedTuple
)
    # All samples were drawn from the prior -- in this case g(x|x') = g(x) = prior
    # probability of x.
    return DynamicPPL.getlogprior(vi)
end
function log_proposal_density(
    vi::DynamicPPL.OnlyAccsVarInfo,
    strategy::InitFromProposals,
    unspecified_priors::DynamicPPL.VarNamedTuple,
)
    # In this case, the proposal distribution might indeed be conditional, so we need to
    # 'run' the initialisation strategies both ways. Luckily, we don't need to run the model
    # itself, since all the information we need is in the proposals. That is the reason why
    # we have to cache the priors in the InitFromProposals struct -- if any variables were
    # not given an explicit proposal (in `strategy.proposals`) we need to know what their
    # prior was.
    vals = DynamicPPL.get_raw_values(vi)
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
    for (vn, prior) in pairs(unspecified_priors)
        g += logpdf(prior, vals[vn])
    end
    return g
end

# Accumulator to store linked values; but only the ones that have a LinkedRW proposal. Since
# model evaluation should have happened with `s.transform_strategy`, any variables that are
# marked by `s.transform_strategy` as being linked should generate a
# TransformedValue{V,DynamicLink} here.
const MH_ACC_NAME = :MHLinkedValues
function store_linked_values(
    val, tval::DynamicPPL.TransformedValue{V,DynamicPPL.DynamicLink}, logjac, vn, dist
) where {V}
    return DynamicPPL.get_internal_value(tval)
end
function store_linked_values(val, ::DynamicPPL.TransformedValue, logjac, vn, dist)
    return DynamicPPL.DoNotAccumulate()
end
function MHLinkedValuesAccumulator()
    return DynamicPPL.VNTAccumulator{MH_ACC_NAME}(store_linked_values)
end

# Accumulator to store priors for any variables that were not given an explicit proposal.
# This is needed to compute the log-proposal density correctly.
const MH_PRIOR_ACC_NAME = :MHUnspecifiedPriors
struct StoreUnspecifiedPriors
    vns_with_proposal::Set{VarName}
end
function Base.copy(s::StoreUnspecifiedPriors)
    return StoreUnspecifiedPriors(Base.copy(s.vns_with_proposal))
end
function (s::StoreUnspecifiedPriors)(val, tval, logjac, vn, dist::Distribution)
    return if vn in s.vns_with_proposal
        DynamicPPL.DoNotAccumulate()
    else
        dist
    end
end
function MHUnspecifiedPriorsAccumulator(vns_with_proposal)
    return DynamicPPL.VNTAccumulator{MH_PRIOR_ACC_NAME}(
        StoreUnspecifiedPriors(vns_with_proposal)
    )
end

# RWMH can be delegated to AdvancedMH. The type bound is intentionally lax because we just
# let the MvNormal constructor handle it.
function MH(cov_matrix::Any)
    return externalsampler(AdvancedMH.RWMH(MvNormal(cov_matrix)); unconstrained=true)
end

####
#### Gibbs interface
####

function gibbs_update_state!!(
    spl::MH,
    state::AbstractVarInfo,
    model::DynamicPPL.Model,
    global_vals::DynamicPPL.VarNamedTuple,
)
    # `state` here is a AbstractVarInfo; the MH sampler since Turing v0.40 only uses
    # the accumulator part of the state. We do need to reevaluate the model though
    # because it's necessary for the log-probability to be updated to reflect the new
    # values in `global_vals`. If we don't do that, the MH acceptance step will return
    # wrong results.
    #
    # In order to make sure that our logjacs are consistent with what the sampler expects,
    # we can use `spl.transform_strategy` here.
    #
    # TODO(penelopeysm): Is the `nothing` fallback OK, or do we need InitFromPrior as
    # a fallback? In the latter case, how do we control `rng`?
    init_strat = DynamicPPL.InitFromParams(global_vals, nothing)
    return last(DynamicPPL.init!!(model, state, init_strat, spl.transform_strategy))
end
