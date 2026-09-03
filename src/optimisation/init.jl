using DynamicPPL: AbstractInitStrategy, AbstractAccumulator
using Distributions

# Store model-shaped constraints separately from keys retained only for reachability errors.
struct ModelConstraints{V<:VarNamedTuple}
    values::V
    unmatched::Vector{Pair{VarName,Any}}
end
ModelConstraints(values::VarNamedTuple) = ModelConstraints(values, Pair{VarName,Any}[])

function _model_constraints_from_pairs(constraints, template::VarNamedTuple)
    values = VarNamedTuple()
    unmatched = Pair{VarName,Any}[]
    for (vn, value) in constraints
        sym = AbstractPPL.getsym(vn)
        if !haskey(template.data, sym)
            push!(unmatched, vn => value)
            continue
        end
        try
            values = DynamicPPL.templated_setindex!!(values, value, vn, template.data[sym])
        catch err
            err isa BoundsError || rethrow()
            # Preserve out-of-range keys for the domain-specific coverage check below.
            push!(unmatched, vn => value)
        end
    end
    return ModelConstraints(values, unmatched)
end
function ModelConstraints(constraints::AbstractDict{<:VarName}, template::VarNamedTuple)
    return _model_constraints_from_pairs(constraints, template)
end
function ModelConstraints(constraints::NamedTuple, ::VarNamedTuple)
    return ModelConstraints(VarNamedTuple(constraints))
end
function ModelConstraints(constraints::VarNamedTuple, template::VarNamedTuple)
    return _model_constraints_from_pairs(pairs(constraints), template)
end

function _constraint_pairs(constraints::ModelConstraints)
    return Iterators.flatten((pairs(constraints.values), constraints.unmatched))
end

"""
    InitWithConstraintCheck(lb, ub, actual_strategy) <: AbstractInitStrategy

Initialise parameters with `actual_strategy`, but check that the initialised
parameters satisfy any bounds in `lb` and `ub`.
"""
struct InitWithConstraintCheck{Tlb<:ModelConstraints,Tub<:ModelConstraints} <:
       AbstractInitStrategy
    lb::Tlb
    ub::Tub
    actual_strategy::AbstractInitStrategy
end

function get_constraints(constraints::VarNamedTuple, vn::VarName)
    if haskey(constraints, vn)
        return constraints[vn]
    else
        return nothing
    end
end
function get_constraints(constraints::ModelConstraints, vn::VarName)
    return get_constraints(constraints.values, vn)
end

const MAX_ATTEMPTS = 1000

"""
    satisfies_constraints(lb, ub, proposed_val, dist)

Check whether `proposed_val` satisfies the constraints defined by `lb` and `ub`.

The methods that this function provides therefore dictate what values users can specify for
different types of distributions. For example, for `UnivariateDistribution`, the constraints
must be supplied as `Real` numbers. If other kinds of constraints are given, it will hit the
fallback method and an error will be thrown.

This method intentionally does not handle `NaN` values as that is left to the optimiser to
deal with.
"""
function satisfies_constraints(
    lb::Union{Nothing,Real},
    ub::Union{Nothing,Real},
    proposed_val::Real,
    ::UnivariateDistribution,
)
    satisfies_lb = lb === nothing || proposed_val >= lb
    satisfies_ub = ub === nothing || proposed_val <= ub
    return isnan(proposed_val) || (satisfies_lb && satisfies_ub)
end
# x may be nothing so we need to take care of that
_prevfloat(x::AbstractFloat) = prevfloat(x)
_prevfloat(x::AbstractArray{<:AbstractFloat}) = map(prevfloat, x)
_prevfloat(x) = x
_nextfloat(x::AbstractFloat) = nextfloat(x)
_nextfloat(x::AbstractArray{<:AbstractFloat}) = map(nextfloat, x)
_nextfloat(x) = x
function satisfies_constraints(
    lb::Union{Nothing,AbstractFloat},
    ub::Union{Nothing,AbstractFloat},
    proposed_val::ForwardDiff.Dual,
    ::UnivariateDistribution,
)
    # The prevfloat/nextfloat is needed because ForwardDiff.Dual(2.0, 1.0) > 2.0 returns
    # true, even though the primal value is within the constraints.
    satisfies_lb = lb === nothing || proposed_val > prevfloat(lb)
    satisfies_ub = ub === nothing || proposed_val < nextfloat(ub)
    return isnan(proposed_val) || (satisfies_lb && satisfies_ub)
end
function satisfies_constraints(
    lb::Union{Nothing,AbstractArray{<:Real}},
    ub::Union{Nothing,AbstractArray{<:Real}},
    proposed_val::AbstractArray{<:Real},
    ::Union{MultivariateDistribution,MatrixDistribution},
)
    satisfies_lb =
        lb === nothing || all(p -> isnan(p[1]) || p[1] >= p[2], zip(proposed_val, lb))
    satisfies_ub =
        ub === nothing || all(p -> isnan(p[1]) || p[1] <= p[2], zip(proposed_val, ub))
    return satisfies_lb && satisfies_ub
end
function satisfies_constraints(
    lb::Union{Nothing,AbstractArray{<:AbstractFloat}},
    ub::Union{Nothing,AbstractArray{<:AbstractFloat}},
    proposed_val::AbstractArray{<:ForwardDiff.Dual},
    dist::Union{MultivariateDistribution,MatrixDistribution},
)
    satisfies_lb =
        lb === nothing ||
        all(p -> isnan(p[1]) || p[1] > prevfloat(p[2]), zip(proposed_val, lb))
    satisfies_ub =
        ub === nothing ||
        all(p -> isnan(p[1]) || p[1] < nextfloat(p[2]), zip(proposed_val, ub))
    return satisfies_lb && satisfies_ub
end
function satisfies_constraints(
    lb::Union{Nothing,NamedTuple},
    ub::Union{Nothing,NamedTuple},
    proposed_val::NamedTuple,
    dist::Distributions.ProductNamedTupleDistribution,
)
    for sym in keys(proposed_val)
        this_lb = lb === nothing ? nothing : get(lb, sym, nothing)
        this_ub = ub === nothing ? nothing : get(ub, sym, nothing)
        this_val = proposed_val[sym]
        this_dist = dist.dists[sym]
        if !satisfies_constraints(this_lb, this_ub, this_val, this_dist)
            return false
        end
    end
    return true
end
function satisfies_constraints(lb::Any, ub::Any, ::Any, d::Distribution)
    # Trivially satisfied if no constraints are given.
    lb === nothing && ub === nothing && return true
    # Otherwise
    throw(
        ArgumentError(
            "Constraints of type $((typeof(lb), typeof(ub))) are not yet implemented for distribution $(typeof(d)). If you need this functionality, please open an issue at https://github.com/TuringLang/Turing.jl/issues.",
        ),
    )
end

function DynamicPPL.init(
    rng::Random.AbstractRNG, vn::VarName, dist::Distribution, c::InitWithConstraintCheck
)
    # First check that the constraints are sensible. The call to satisfies_constraints will
    # error if `lb` is 'greater' than `ub`.
    lb = get_constraints(c.lb, vn)
    ub = get_constraints(c.ub, vn)
    if lb !== nothing && ub !== nothing && !satisfies_constraints(lb, ub, lb, dist)
        throw(ArgumentError("Lower bound for variable $(vn) is greater than upper bound."))
    end
    # The inner `init` might (for whatever reason) return linked or otherwise
    # transformed values. We need to transform them back into to unlinked space,
    # so that we can check the constraints properly.
    tv = DynamicPPL.init(rng, vn, dist, c.actual_strategy)
    proposed_val = DynamicPPL.get_raw_value(tv, dist)
    attempts = 1
    while !satisfies_constraints(lb, ub, proposed_val, dist)
        if attempts >= MAX_ATTEMPTS
            throw(
                ArgumentError(
                    "Could not initialise variable $(vn) within constraints after $(MAX_ATTEMPTS) attempts; please supply your own initialisation values using `InitFromParams`, or check that the values you supplied are valid",
                ),
            )
        end
        tv = DynamicPPL.init(rng, vn, dist, c.actual_strategy)
        proposed_val = DynamicPPL.get_raw_value(tv, dist)
        attempts += 1
    end
    return tv
end

can_have_linked_constraints(::Distribution) = false
can_have_linked_constraints(::UnivariateDistribution) = true
can_have_linked_constraints(::MultivariateDistribution) = true
can_have_linked_constraints(::MatrixDistribution) = false
function can_have_linked_constraints(pd::Distributions.Product)
    return all(can_have_linked_constraints.(pd.v))
end
function can_have_linked_constraints(pd::Distributions.ProductDistribution)
    return all(can_have_linked_constraints.(pd.dists))
end
function can_have_linked_constraints(pd::Distributions.ProductNamedTupleDistribution)
    return all(can_have_linked_constraints.(values(pd.dists)))
end
can_have_linked_constraints(::Dirichlet) = false
can_have_linked_constraints(::LKJCholesky) = false

struct ConstraintAccumulator{
    T<:DynamicPPL.AbstractTransformStrategy,Vlb<:ModelConstraints,Vub<:ModelConstraints
} <: AbstractAccumulator
    "Whether to store constraints in linked space or not."
    transform_strategy::T
    "A mapping of VarNames to lower bounds in untransformed space."
    lb::Vlb
    "A mapping of VarNames to upper bounds in untransformed space."
    ub::Vub
    "The initial values for the optimisation in linked space (if link=true) or unlinked
    space (if link=false)."
    init_vecs::Dict{VarName,AbstractVector}
    "The lower bound vectors for the optimisation in linked space (if link=true) or unlinked
    space (if link=false)."
    lb_vecs::Dict{VarName,AbstractVector}
    "The upper bound vectors for the optimisation in linked space (if link=true) or unlinked
    space (if link=false)."
    ub_vecs::Dict{VarName,AbstractVector}
    function ConstraintAccumulator(
        link::DynamicPPL.AbstractTransformStrategy,
        lb::ModelConstraints,
        ub::ModelConstraints,
    )
        return new{typeof(link),typeof(lb),typeof(ub)}(
            link,
            lb,
            ub,
            Dict{VarName,AbstractVector}(),
            Dict{VarName,AbstractVector}(),
            Dict{VarName,AbstractVector}(),
        )
    end
end
const CONSTRAINT_ACC_NAME = :OptimConstraints
DynamicPPL.accumulator_name(::ConstraintAccumulator) = CONSTRAINT_ACC_NAME
function DynamicPPL.accumulate_assume!!(
    acc::ConstraintAccumulator,
    val::Any,
    tval::Any,
    logjac::Any,
    vn::VarName,
    dist::Distribution,
    template::Any,
)
    # First check if we have any incompatible constraints + linking. 'Incompatible', here,
    # means that the constraints as defined in the unlinked space do not map to box
    # constraints in the linked space, which would make it impossible to generate
    # appropriate `lb` and `ub` arguments to pass to Optimization.jl. This is the case for
    # e.g. Dirichlet.
    lb = get_constraints(acc.lb, vn)
    ub = get_constraints(acc.ub, vn)
    should_be_linked =
        DynamicPPL.target_transform(acc.transform_strategy, vn) isa DynamicPPL.DynamicLink
    if (lb !== nothing || ub !== nothing) &&
        should_be_linked &&
        !can_have_linked_constraints(dist)
        throw(
            ArgumentError(
                "Cannot use constraints for variable $(vn) with distribution $(typeof(dist)) when performing linked optimisation; this is because the constraints cannot be cleanly mapped to linked space. If you need to use constraints for this variable, please set `link=false` when optimising, or manually perform optimisation with your own LogDensityFunction.",
            ),
        )
    end
    target_tfm = DynamicPPL.target_transform(acc.transform_strategy, vn)
    transform_fn = if target_tfm isa DynamicPPL.DynamicLink
        Bijectors.VectorBijectors.to_linked_vec(dist)
    elseif target_tfm isa DynamicPPL.Unlink
        Bijectors.VectorBijectors.to_vec(dist)
    elseif target_tfm isa DynamicPPL.FixedTransform
        Bijectors.inverse(target_tfm.transform)
    else
        error(
            "don't know how to handle transform strategy $(acc.transform_strategy) for variable $(vn)",
        )
    end
    # Transform the value and store it.
    vectorised_val = transform_fn(val)
    if !(vectorised_val isa AbstractVector)
        error(
            "The transform strategy used ($(acc.transform_strategy)) generated a value for variable $(vn) that is not a vector; in general this cannot be handled by Turing",
        )
    end
    acc.init_vecs[vn] = vectorised_val
    nelems = length(vectorised_val)
    # Then generate the constraints using the same transform.
    if lb !== nothing
        acc.lb_vecs[vn] = transform_fn(lb)
    else
        acc.lb_vecs[vn] = fill(-Inf, nelems)
    end
    if ub !== nothing
        acc.ub_vecs[vn] = transform_fn(ub)
    else
        acc.ub_vecs[vn] = fill(Inf, nelems)
    end
    return acc
end
function DynamicPPL.accumulate_observe!!(
    acc::ConstraintAccumulator, ::Distribution, ::Any, ::Union{VarName,Nothing}, ::Any
)
    return acc
end
function DynamicPPL.reset(acc::ConstraintAccumulator)
    return ConstraintAccumulator(acc.transform_strategy, acc.lb, acc.ub)
end
function Base.copy(acc::ConstraintAccumulator)
    # ConstraintAccumulator should not ever modify `acc.lb` or `acc.ub` (and when
    # constructing it inside `make_optim_bounds_and_init` we make sure to deepcopy any user
    # input), so there is no chance that `lb` or `ub` could ever be mutated once they're
    # inside the accumulator. Hence we don't need to copy them.
    return ConstraintAccumulator(acc.transform_strategy, acc.lb, acc.ub)
end
function DynamicPPL.split(acc::ConstraintAccumulator)
    return ConstraintAccumulator(acc.transform_strategy, acc.lb, acc.ub)
end
function DynamicPPL.combine(acc1::ConstraintAccumulator, acc2::ConstraintAccumulator)
    combined = ConstraintAccumulator(acc1.transform_strategy, acc1.lb, acc1.ub)
    combined.init_vecs = merge(acc1.init_vecs, acc2.init_vecs)
    combined.lb_vecs = merge(acc1.lb_vecs, acc2.lb_vecs)
    combined.ub_vecs = merge(acc1.ub_vecs, acc2.ub_vecs)
    return combined
end

"""
    make_optim_bounds_and_init(
        rng::Random.AbstractRNG,
        ldf::LogDensityFunction,
        initial_params::AbstractInitStrategy,
        lb::VarNamedTuple,
        ub::VarNamedTuple,
    )

Generate a tuple of `(lb_vec, ub_vec, init_vec)` which are suitable for directly passing to
Optimization.jl. All three vectors returned will be in the unlinked or linked space
depending on `ldf.transform_strategy`, which in turn is defined by the value of `link` passed
to `mode_estimate`.

The `lb` and `ub` arguments, as well as any `initial_params` provided as `InitFromParams`,
are expected to be in the unlinked space.
"""
function make_optim_bounds_and_init(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction,
    initial_params::AbstractInitStrategy,
    lb::VarNamedTuple,
    ub::VarNamedTuple,
)
    return make_optim_bounds_and_init(
        rng,
        ldf,
        initial_params,
        ModelConstraints(lb),
        ModelConstraints(ub),
        Dict{VarName,Any}(),
        Dict{VarName,Any}(),
    )
end

function make_optim_bounds_and_init(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction,
    initial_params::AbstractInitStrategy,
    lb::ModelConstraints,
    ub::ModelConstraints,
    resolved_lb::AbstractDict,
    resolved_ub::AbstractDict,
)
    # Initialise a VarInfo with parameters that satisfy the constraints.
    # ConstraintAccumulator only needs the raw value so we can use UnlinkAll() as the
    # transform strategy for this
    init_strategy = InitWithConstraintCheck(lb, ub, initial_params)
    vi = DynamicPPL.OnlyAccsVarInfo((
        ConstraintAccumulator(ldf.transform_strategy, deepcopy(lb), deepcopy(ub)),
    ))
    _, vi = DynamicPPL.init!!(rng, ldf.model, vi, init_strategy, DynamicPPL.UnlinkAll())
    # Now extract the accumulator, and construct the vectorised constraints using the
    # ranges stored in the LDF.
    constraint_acc = DynamicPPL.getacc(vi, Val(CONSTRAINT_ACC_NAME))
    nelems = LogDensityProblems.dimension(ldf)
    # TODO(penelopeysm) This should really be exported
    et = eltype(DynamicPPL.get_input_vector_type(ldf))
    inits = fill(et(NaN), nelems)
    lb_vec = fill(et(-Inf), nelems)
    ub_vec = fill(et(Inf), nelems)
    # Which variables a bound was actually written for. `check_constraints_reached` needs this
    # rather than a prediction of it: asking `get_constraints` instead reported a bound as used
    # whenever the collection could answer for the variable, which is not the same as the bound
    # reaching this assembly. A bound naming non-leading elements of a variable the model writes
    # whole -- `x[2]`, or `x[1]` and `x[3]` -- never arrives here at all, and was passed as
    # harmless while the mode came back unconstrained.
    applied_lb, applied_ub = VarName[], VarName[]
    for (vn, init_val) in constraint_acc.init_vecs
        range = DynamicPPL.get_range_and_transform(ldf, vn).range
        inits[range] = init_val
        if haskey(constraint_acc.lb_vecs, vn)
            check_bound_covers(lb, "lb", vn, range)
            lb_vec[range] = constraint_acc.lb_vecs[vn]
            # Recorded as applied only if the caller's value can be read for this variable.
            # `haskey` is true whenever the accumulator visited it, which it does even when the
            # value cannot be read -- a scalar `lb = (x = 0.9,)` against an element-wise
            # `x[1] ~` leaves an infinite bound behind, and counting that as applied dropped
            # the caller's bound in silence. Testing `isfinite` on the result instead would
            # refuse a bound the caller deliberately wrote as infinite.
            site_lb = get_constraints(lb, vn)
            if site_lb !== nothing
                push!(applied_lb, vn)
                resolved_lb[vn] = site_lb
            end
        end
        if haskey(constraint_acc.ub_vecs, vn)
            check_bound_covers(ub, "ub", vn, range)
            ub_vec[range] = constraint_acc.ub_vecs[vn]
            site_ub = get_constraints(ub, vn)
            if site_ub !== nothing
                push!(applied_ub, vn)
                resolved_ub[vn] = site_ub
            end
        end
    end
    # The loop above visits the model's variables, so a bound whose key names none of them is
    # never consulted and the mode comes back unconstrained. Name it instead.
    check_constraints_reached(lb, "lb", applied_lb, keys(constraint_acc.init_vecs))
    check_constraints_reached(ub, "ub", applied_ub, keys(constraint_acc.init_vecs))
    # Make sure we have filled in all values. This should never happen, but we should just
    # check.
    if any(isnan, inits)
        error("Could not generate vector of initial values as some values are missing.")
    end
    # Concretise before returning.
    return [x for x in lb_vec], [x for x in ub_vec], [x for x in inits]
end

"""
    check_bound_covers(constraints, name, vn, range)

Throw unless `constraints` bounds every element of `vn`, or none of them.

A bound reaches the optimiser only if it covers a whole variable as the model writes it, so a
bound on part of one is a mistake rather than a partial constraint: the unmentioned elements are
left free and the mode comes back unconstrained in them.

Scalar leaves are counted, not keys, because one key can hold many: `lb = (x = [0.9, 0.9],)`
covers two under a single key, `Dict(x[1] => 0.9, x[2] => 0.9)` the same two under one key each,
and `lb = (x = (a = [0.1, 0.1], b = 0.1),)` covers three. A key coarser than `vn` counts only if
a value can be read for `vn` -- a scalar `lb = (x = 0.9,)` holds nothing for an element-wise
`x[1] ~` and is never applied.
"""
function check_bound_covers(constraints::ModelConstraints, name, vn, range)
    covered = 0
    for (key, value) in _constraint_pairs(constraints)
        if AbstractPPL.subsumes(vn, key)
            # A key inside `vn` contributes the scalar leaves it holds, not the length of its
            # value: `(a = [0.1, 0.1], b = 0.1)` is one key of length two holding three
            # elements.
            covered += count(Returns(true), DynamicPPL.varname_leaves(key, value))
        elseif AbstractPPL.subsumes(key, vn)
            # A key coarser than `vn` covers it only if a value can actually be read for `vn`:
            # `lb = (x = [0.9, 0.9],)` bounds both of an element-wise `x[1] ~`, `x[2] ~`, while
            # a scalar `lb = (x = 0.9,)` holds nothing for `x[1]` and is never applied. Asking
            # subsumption alone treated the scalar as covering them and dropped it in silence.
            get_constraints(constraints, vn) === nothing || return nothing
        end
    end
    # Nothing bound for this variable at all is not this check's business:
    # `check_constraints_reached` decides whether such a key is malformed or merely moot.
    (covered == 0 || covered == length(range)) && return nothing
    return throw(
        ArgumentError(
            "`$(name)` names $(covered) element(s) under $(vn), which the model writes as one " *
            "value of $(length(range)) element(s). A bound is honoured only if it covers the " *
            "whole variable, so bound every element of $(vn) or none of it.",
        ),
    )
end

"""
    check_constraints_reached(constraints, name, applied, model_vns)

Complain about any bound that was not applied.

`applied` lists the variables a bound was actually written for, taken from the assembly rather
than predicted: whether the collection *could* answer for a variable is not the same question as
whether a bound reached the optimiser.

`applied` and `model_vns` hold whole tilde `VarName`s, which may be compound, so a bound is
matched against them by subsumption in either direction. That is why the accounting is per leaf
and not per key: `keys(constraints)` gives the caller's key, `x` for `lb = (x = [0.9, 100.0],)`,
and subsumption would let the one element a model writing only `x[1]` consumes stand for the
whole key, dropping the rest without a word.

An unapplied bound is fatal when the model reaches a variable covering it, since bounding part of
a value the model writes whole cannot be honoured. When no reached variable covers it the bound
is merely moot -- the key may name a variable that is conditioned, fixed, or in a branch not
taken, which is indistinguishable here from one naming nothing -- so that warns.
"""
function check_constraints_reached(constraints::ModelConstraints, name, applied, model_vns)
    claims(vns, leaf) =
        any(vns) do vn
            AbstractPPL.subsumes(vn, leaf) || AbstractPPL.subsumes(leaf, vn)
        end
    leaves = Iterators.flatten(
        DynamicPPL.varname_leaves(key, value) for
        (key, value) in _constraint_pairs(constraints)
    )
    unapplied = [leaf for leaf in leaves if !claims(applied, leaf)]
    isempty(unapplied) && return nothing
    malformed = filter(leaf -> claims(model_vns, leaf), unapplied)
    if !isempty(malformed)
        # Name the variables that could have used these keys, not every variable in the model.
        covering = sort!(unique(vn for vn in model_vns if claims(malformed, vn)); by=string)
        throw(
            ArgumentError(
                "`$(name)` has bounds for $(join(sort!(malformed; by=string), ", ")) that " *
                "cannot be applied. The model writes those values as " *
                "$(join(covering, ", ")), and a bound is honoured only if it covers a whole " *
                "variable as the model writes it. Bound every element of it, or none.",
            ),
        )
    end
    @warn "`$(name)` has bounds for $(join(unapplied, ", ")) that no variable of the model " *
        "can use, so they have no effect. The model's variables are $(join(model_vns, ", ")). " *
        "This is harmless if the variable is conditioned, fixed, or in a branch not taken."
    return nothing
end
