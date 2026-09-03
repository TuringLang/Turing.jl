using DynamicPPL: AbstractInitStrategy, AbstractAccumulator
using Distributions

"""
    InitWithConstraintCheck(lb, ub, actual_strategy) <: AbstractInitStrategy

Initialise parameters with `actual_strategy`, but check that the initialised
parameters satisfy any bounds in `lb` and `ub`.
"""
struct InitWithConstraintCheck{Tlb<:VarNamedTuple,Tub<:VarNamedTuple} <:
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
    T<:DynamicPPL.AbstractTransformStrategy,Vlb<:VarNamedTuple,Vub<:VarNamedTuple
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
        link::DynamicPPL.AbstractTransformStrategy, lb::VarNamedTuple, ub::VarNamedTuple
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
    for (vn, init_val) in constraint_acc.init_vecs
        range = DynamicPPL.get_range_and_transform(ldf, vn).range
        inits[range] = init_val
        if haskey(constraint_acc.lb_vecs, vn)
            lb_vec[range] = check_bound_length(constraint_acc.lb_vecs[vn], "lb", vn, range)
        end
        if haskey(constraint_acc.ub_vecs, vn)
            ub_vec[range] = check_bound_length(constraint_acc.ub_vecs[vn], "ub", vn, range)
        end
    end
    # The loop above visits the model's variables, so a bound whose key names none of them is
    # never consulted and the mode comes back unconstrained. Name it instead.
    check_constraints_reached(lb, "lb", keys(constraint_acc.init_vecs))
    check_constraints_reached(ub, "ub", keys(constraint_acc.init_vecs))
    # Make sure we have filled in all values. This should never happen, but we should just
    # check.
    if any(isnan, inits)
        error("Could not generate vector of initial values as some values are missing.")
    end
    # Concretise before returning.
    return [x for x in lb_vec], [x for x in ub_vec], [x for x in inits]
end

"""
    check_bound_length(bound, name, vn, range)

Return `bound`, having checked it has one entry per element of `vn`.

A bound naming part of a variable the model writes whole -- `lb = Dict(@varname(x[1]) => 0.0)`
against `x ~ MvNormal(zeros(2), I)` -- answers `haskey` and so reaches this assembly, but with
fewer entries than the variable occupies. Assigning it raised a bare `DimensionMismatch` naming
neither the variable nor the keyword.
"""
function check_bound_length(bound, name, vn, range)
    length(bound) == length(range) && return bound
    throw(
        ArgumentError(
            "`$(name)` gives $(length(bound)) bound(s) for $(vn), which the model writes as " *
            "one value of $(length(range)) element(s). Bound every element of $(vn) or none " *
            "of it.",
        ),
    )
end

"""
    check_constraints_reached(constraints::VarNamedTuple, name, model_vns)

Warn about any bound in `constraints` that no variable of the model can use.

Bounds are applied by asking `get_constraints(constraints, vn)` for each variable the model
reaches, so one it cannot answer for is ignored, and `estimate_mode` returns as if that bound had
never been given. The original silence over that was the defect; `lb = (nope = 0.0,)` alone left
`bounds_kwargs` empty and the mode came back unconstrained.

Two granularities have to be reconciled, and getting that wrong is what made an earlier version
of this check refuse bounds that work. `keys(constraints)` enumerates *leaves* -- a bound written
`Dict(@varname(x[1:2]) => [0.9, 0.9])` enumerates as `x[1]`, `x[2]` -- while `model_vns` holds
whole tilde `VarName`s, which may be compound. Asking about a single leaf in isolation therefore
answers `false` for the compound name that in fact consumes it. So a leaf counts as used when
some reached variable both draws from these constraints and stands in a subsumption relation to
it, which compares the two at the same granularity.

This warns rather than throws because a bound naming a variable the model has but does not reach
-- conditioned, fixed, or in a branch not taken -- is indistinguishable here from one naming
nothing at all: `model_vns` holds only what was reached. Refusing it broke the ordinary pattern
of reusing one `lb`/`ub` across conditioned variants of a model, whose modes were correctly
constrained on the variables that do exist. A bound of the wrong *shape* is a different matter
and still throws; see [`check_bound_length`](@ref).
"""
function check_constraints_reached(constraints::VarNamedTuple, name, model_vns)
    used(leaf) =
        any(model_vns) do vn
            get_constraints(constraints, vn) === nothing && return false
            return AbstractPPL.subsumes(vn, leaf) || AbstractPPL.subsumes(leaf, vn)
        end
    unused = [leaf for leaf in keys(constraints) if !used(leaf)]
    isempty(unused) && return nothing
    @warn "`$(name)` has bounds for $(join(unused, ", ")) that no variable of the model can " *
        "use, so they have no effect. The model's variables are $(join(model_vns, ", ")). A " *
        "bound is applied only if it covers a whole variable as the model writes it; this is " *
        "harmless if the variable is conditioned, fixed, or in a branch not taken."
    return nothing
end
