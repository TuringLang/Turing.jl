using DynamicPPL: AbstractInitStrategy, AbstractAccumulator
using Distributions

# TODO(penelopeysm): Eventually replace with VNT
const NTOrVNDict = Union{NamedTuple,AbstractDict{<:VarName,<:Any}}
function get_constraints(bounds::NTOrVNDict, vn::VarName)
    return if AbstractPPL.hasvalue(bounds, vn)
        return AbstractPPL.getvalue(bounds, vn)
    else
        nothing
    end
end

"""
    InitWithConstraintCheck(lb, ub, actual_strategy) <: AbstractInitStrategy

Initialise parameters with `actual_strategy`, but check that the initialised
parameters satisfy any bounds in `lb` and `ub`.
"""
struct InitWithConstraintCheck{Tlb<:NTOrVNDict,Tub<:NTOrVNDict} <: AbstractInitStrategy
    lb::Tlb
    ub::Tub
    actual_strategy::AbstractInitStrategy
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
function satisfies_constraints(
    lb::Union{Nothing,Real},
    ub::Union{Nothing,Real},
    proposed_val::ForwardDiff.Dual,
    dist::UnivariateDistribution,
)
    # This overload is needed because ForwardDiff.Dual(2.0, 1.0) > 2.0 returns true, even
    # though the primal value is within the constraints.
    return satisfies_constraints(lb, ub, ForwardDiff.value(proposed_val), dist)
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
    lb::Union{Nothing,AbstractArray{<:Real}},
    ub::Union{Nothing,AbstractArray{<:Real}},
    proposed_val::AbstractArray{<:ForwardDiff.Dual},
    dist::Union{MultivariateDistribution,MatrixDistribution},
)
    return satisfies_constraints(lb, ub, ForwardDiff.value.(proposed_val), dist)
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
    maybe_transformed_val = DynamicPPL.init(rng, vn, dist, c.actual_strategy)
    proposed_val = DynamicPPL.get_true_value(maybe_transformed_val)
    attempts = 1
    while !satisfies_constraints(lb, ub, proposed_val, dist)
        if attempts >= MAX_ATTEMPTS
            throw(
                ArgumentError(
                    "Could not initialise variable $(vn) within constraints after $(MAX_ATTEMPTS) attempts; please supply your own initialisation values using `InitFromParams`, or check that the values you supplied are valid",
                ),
            )
        end
        maybe_transformed_val = DynamicPPL.init(rng, vn, dist, c.actual_strategy)
        proposed_val = DynamicPPL.get_true_value(maybe_transformed_val)
        attempts += 1
    end
    return DynamicPPL.UntransformedValue(proposed_val)
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

struct ConstraintAccumulator <: AbstractAccumulator
    "Whether to store constraints in linked space or not."
    link::Bool
    "A mapping of VarNames to lower bounds in untransformed space."
    lb::NTOrVNDict
    "A mapping of VarNames to upper bounds in untransformed space."
    ub::NTOrVNDict
    "The initial values for the optimisation in linked space (if link=true) or unlinked
    space (if link=false)."
    init_vecs::Dict{VarName,AbstractVector}
    "The lower bound vectors for the optimisation in linked space (if link=true) or unlinked
    space (if link=false)."
    lb_vecs::Dict{VarName,AbstractVector}
    "The upper bound vectors for the optimisation in linked space (if link=true) or unlinked
    space (if link=false)."
    ub_vecs::Dict{VarName,AbstractVector}
    function ConstraintAccumulator(link::Bool, lb::NTOrVNDict, ub::NTOrVNDict)
        return new(
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
    if (lb !== nothing || ub !== nothing) && acc.link && !can_have_linked_constraints(dist)
        throw(
            ArgumentError(
                "Cannot use constraints for variable $(vn) with distribution $(typeof(dist)) when performing linked optimisation; this is because the constraints cannot be cleanly mapped to linked space. If you need to use constraints for this variable, please set `link=false` when optimising, or manually perform optimisation with your own LogDensityFunction.",
            ),
        )
    end
    transform = if acc.link
        DynamicPPL.to_linked_vec_transform(dist)
    else
        DynamicPPL.to_vec_transform(dist)
    end
    # Transform the value and store it.
    vectorised_val = transform(val)
    acc.init_vecs[vn] = vectorised_val
    nelems = length(vectorised_val)
    # Then generate the constraints using the same transform.
    if lb !== nothing
        acc.lb_vecs[vn] = transform(lb)
    else
        acc.lb_vecs[vn] = fill(-Inf, nelems)
    end
    if ub !== nothing
        acc.ub_vecs[vn] = transform(ub)
    else
        acc.ub_vecs[vn] = fill(Inf, nelems)
    end
    return acc
end
function DynamicPPL.accumulate_observe!!(
    acc::ConstraintAccumulator, ::Distribution, ::Any, ::Union{VarName,Nothing}
)
    return acc
end
function DynamicPPL.reset(acc::ConstraintAccumulator)
    return ConstraintAccumulator(acc.link, acc.lb, acc.ub)
end
function Base.copy(acc::ConstraintAccumulator)
    # ConstraintAccumulator should not ever modify `acc.lb` or `acc.ub` (and when
    # constructing it inside `make_optim_bounds_and_init` we make sure to deepcopy any user
    # input), so there is no chance that `lb` or `ub` could ever be mutated once they're
    # inside the accumulator. Hence we don't need to copy them.
    return ConstraintAccumulator(acc.link, acc.lb, acc.ub)
end
function DynamicPPL.split(acc::ConstraintAccumulator)
    return ConstraintAccumulator(acc.link, acc.lb, acc.ub)
end
function DynamicPPL.combine(acc1::ConstraintAccumulator, acc2::ConstraintAccumulator)
    combined = ConstraintAccumulator(acc1.link, acc1.lb, acc1.ub)
    combined.init_vecs = merge(acc1.init_vecs, acc2.init_vecs)
    combined.lb_vecs = merge(acc1.lb_vecs, acc2.lb_vecs)
    combined.ub_vecs = merge(acc1.ub_vecs, acc2.ub_vecs)
    return combined
end

function _get_ldf_range(ldf::LogDensityFunction, vn::VarName)
    if haskey(ldf._varname_ranges, vn)
        return ldf._varname_ranges[vn].range
    elseif haskey(ldf._iden_varname_ranges, AbstractPPL.getsym(vn))
        return ldf._iden_varname_ranges[AbstractPPL.getsym(vn)].range
    else
        # Should not happen.
        error("could not find range for variable name $(vn) in LogDensityFunction")
    end
end

"""
    make_optim_bounds_and_init(
        rng::Random.AbstractRNG,
        ldf::LogDensityFunction{Tlink},
        initial_params::AbstractInitStrategy,
        lb::NTOrVNDict,
        ub::NTOrVNDict,
    ) where {Tlink}

Generate a tuple of `(lb_vec, ub_vec, init_vec)` which are suitable for directly passing to
Optimization.jl. All three vectors returned will be in the unlinked or linked space
depending on the value of `link`.

The `lb` and `ub` arguments, as well as any `initial_params` provided as `InitFromParams`,
are expected to be in the unlinked space.
"""
function make_optim_bounds_and_init(
    rng::Random.AbstractRNG,
    ldf::LogDensityFunction{Tlink},
    initial_params::AbstractInitStrategy,
    lb::NTOrVNDict,
    ub::NTOrVNDict,
) where {Tlink}
    # Initialise a VarInfo with parameters that satisfy the constraints.
    ctx = DynamicPPL.InitContext(rng, InitWithConstraintCheck(lb, ub, initial_params))
    new_model = DynamicPPL.setleafcontext(ldf.model, ctx)
    # Run the model.
    vi = DynamicPPL.OnlyAccsVarInfo((
        ConstraintAccumulator(Tlink, deepcopy(lb), deepcopy(ub)),
    ))
    _, vi = DynamicPPL.evaluate!!(new_model, vi)
    # Now extract the accumulator, and construct the vectorised constraints using the
    # ranges stored in the LDF.
    constraint_acc = DynamicPPL.getacc(vi, Val(CONSTRAINT_ACC_NAME))
    nelems = LogDensityProblems.dimension(ldf)
    inits = fill(NaN, nelems)
    lb = fill(-Inf, nelems)
    ub = fill(Inf, nelems)
    for (vn, init_val) in constraint_acc.init_vecs
        range = _get_ldf_range(ldf, vn)
        inits[range] = init_val
        if haskey(constraint_acc.lb_vecs, vn)
            lb[range] = constraint_acc.lb_vecs[vn]
        end
        if haskey(constraint_acc.ub_vecs, vn)
            ub[range] = constraint_acc.ub_vecs[vn]
        end
    end
    # Make sure we have filled in all values. This should never happen, but we should just
    # check.
    if any(isnan, inits)
        error("Could not generate vector of initial values as some values are missing.")
    end
    # Concretise before returning.
    return [x for x in lb], [x for x in ub], [x for x in inits]
end
