module Optimisation

using ..Turing
using AbstractPPL: AbstractPPL, VarName
using DynamicPPL: DynamicPPL, VarInfo, LogDensityFunction
using DocStringExtensions: TYPEDFIELDS
using LogDensityProblems: LogDensityProblems
using Optimization: Optimization
using OptimizationOptimJL: LBFGS
using Random: Random
using SciMLBase: SciMLBase
using ADTypes: ADTypes
using StatsBase: StatsBase
using Printf: Printf
using ForwardDiff: ForwardDiff
using StatsAPI: StatsAPI
using Statistics: Statistics
using LinearAlgebra: LinearAlgebra

export maximum_a_posteriori, maximum_likelihood, MAP, MLE, get_vector_params

include("init.jl")

"""
    ModeEstimator

An abstract type to mark whether mode estimation is to be done with maximum a posteriori
(MAP) or maximum likelihood estimation (MLE).
"""
abstract type ModeEstimator end

"""
    MLE <: ModeEstimator

Concrete type for maximum likelihood estimation.
"""
struct MLE <: ModeEstimator end
logprob_func(::MLE) = DynamicPPL.getloglikelihood
logprob_accs(::MLE) = (DynamicPPL.LogLikelihoodAccumulator(),)

"""
    MAP <: ModeEstimator

Concrete type for maximum a posteriori estimation.
"""
struct MAP <: ModeEstimator end
# Note that we use `getlogjoint` rather than `getlogjoint_internal`: this is intentional,
# because even though the VarInfo may be linked, the optimisation target should not take the
# Jacobian term into account.
logprob_func(::MAP) = DynamicPPL.getlogjoint
function logprob_accs(::MAP)
    return (DynamicPPL.LogLikelihoodAccumulator(), DynamicPPL.LogPriorAccumulator())
end

"""
    ModeResult{
        E<:ModeEstimator,
        P<:DynamicPPL.VarNamedTuple,
        LP<:Real,
        L<:DynamicPPL.LogDensityFunction,
        O<:Any,
    }

A wrapper struct to store various results from a MAP or MLE estimation.

## Fields

$(TYPEDFIELDS)
"""
struct ModeResult{
    E<:ModeEstimator,P<:DynamicPPL.VarNamedTuple,LP<:Real,L<:DynamicPPL.LogDensityFunction,O
} <: StatsBase.StatisticalModel
    "The type of mode estimation (MAP or MLE)."
    estimator::E
    "Dictionary of parameter values. These values are always provided in unlinked space,
    even if the optimisation was run in linked space."
    params::P
    "The final log likelihood or log joint, depending on whether `MAP` or `MLE` was run.
    Note that this is the actual log probability of the parameters, i.e., not negated;
    we do need a negated log probability to run the optimisation itself (since it is a
    maximisation), but this is handled in a way that is entirely transparent to the user."
    lp::LP
    "Whether the optimisation was done in a transformed space."
    linked::Bool
    "The LogDensityFunction used to calculate the output. Note that this LogDensityFunction
    calculates the actual (non-negated) log density. It should hold that `m.lp ==
    LogDensityProblems.logdensity(m.ldf, m.optim_result.u)` for a ModeResult `m`.

    The objective function used for minimisation is equivalent to `p ->
    -LogDensityProblems.logdensity(m.ldf, p)`). Note, however, that `p` has to be provided
    as a vector in linked or unlinked space depending on the value of `m.linked`.

    If `m.linked` is true, to evaluate the log-density using unlinked parameters, you
    can use `logjoint(m.ldf.model, params)` where `params` is a NamedTuple or Dictionary
    of unlinked parameters."
    ldf::L
    "The stored optimiser results."
    optim_result::O
end

"""
    ModeResult(
        log_density::DynamicPPL.LogDensityFunction,
        solution::SciMLBase.OptimizationSolution,
        linked::Bool,
        estimator::ModeEstimator,
    )

Create a `ModeResult` for a given `log_density` objective and a `solution` given by `solve`.
The `linked` argument indicates whether the optimization was done in a transformed space.

`Optimization.solve` returns its own result type. This function converts that into the
richer format of `ModeResult`. It also takes care of transforming them back to the original
parameter space in case the optimization was done in a transformed space.
"""
function ModeResult(
    ldf::LogDensityFunction,
    solution::SciMLBase.OptimizationSolution,
    linked::Bool,
    estimator::ModeEstimator,
)
    # Get the parameter values in the original space.
    parameters = DynamicPPL.ParamsWithStats(solution.u, ldf).params
    return ModeResult(estimator, parameters, -solution.objective, linked, ldf, solution)
end

function Base.show(io::IO, ::MIME"text/plain", m::ModeResult)
    printstyled(io, "ModeResult\n"; bold=true)
    # typeof avoids the parentheses in the printed output
    println(io, "  ├ estimator : $(typeof(m.estimator))")
    println(io, "  ├ lp        : $(m.lp)")
    entries = length(m.params) == 1 ? "entry" : "entries"
    println(
        io, "  ├ params    : DynamicPPL.VarNamedTuple with $(length(m.params)) $(entries)"
    )
    for (i, (vn, val)) in enumerate(pairs(m.params))
        tree_char = i == length(m.params) ? "└" : "├"
        println(io, "  │             $(tree_char) $vn => $(val)")
    end
    print(io, "  └ linked    : $(m.linked)")
    return nothing
end

"""
    InitFromParams(
        m::ModeResult,
        fallback::Union{AbstractInitStrategy,Nothing}=InitFromPrior()
    )

Initialize a model from the parameters stored in a `ModeResult`. The `fallback` is used if
some parameters are missing from the `ModeResult`.
"""
function DynamicPPL.InitFromParams(
    m::ModeResult, fallback::Union{DynamicPPL.AbstractInitStrategy,Nothing}=InitFromPrior()
)
    return DynamicPPL.InitFromParams(m.params, fallback)
end

struct ConstraintCheckAccumulator <: AbstractAccumulator
    lb::NTOrVNDict # Must be in unlinked space
    ub::NTOrVNDict # Must be in unlinked space
end
DynamicPPL.accumulator_name(::ConstraintCheckAccumulator) = :OptimConstraintCheck
function DynamicPPL.accumulate_assume!!(
    acc::ConstraintCheckAccumulator,
    val::Any,
    tval::Any,
    logjac::Any,
    vn::VarName,
    dist::Distribution,
    template::Any,
)
    # `val`, `acc.lb`, and `acc.ub` are all in unlinked space.
    lb = get_constraints(acc.lb, vn)
    ub = get_constraints(acc.ub, vn)
    if !satisfies_constraints(lb, ub, val, dist)
        throw(
            DomainError(
                val,
                "\nThe value for variable $(vn) ($(val)) went outside of constraints (lb=$(lb), ub=$(ub)) during optimisation.\n\nThis can happen when using constraints on a variable that has a dynamic support, e.g., `y ~ truncated(Normal(); lower=x)` where `x` is another variable in the model.\n\nTo avoid this, consider either running the optimisation in unlinked space (`link=false`) or removing the constraints.\n\nIf you are sure that this does not matter and you want to suppress this error, you can also set `check_constraints_at_runtime=false`.",
            ),
        )
    end
    return acc
end
function DynamicPPL.accumulate_observe!!(
    acc::ConstraintCheckAccumulator, ::Distribution, ::Any, ::Union{VarName,Nothing}
)
    return acc
end
function DynamicPPL.reset(acc::ConstraintCheckAccumulator)
    return acc
end
function Base.copy(acc::ConstraintCheckAccumulator)
    # The copy here is probably not needed, since lb and ub are never mutated, and we are
    # responsible for generating lb and ub. But we can just `copy` to be safe.
    return ConstraintCheckAccumulator(copy(acc.lb), copy(acc.ub))
end
DynamicPPL.split(acc::ConstraintCheckAccumulator) = acc
DynamicPPL.combine(acc1::ConstraintCheckAccumulator, ::ConstraintCheckAccumulator) = acc1

"""
    estimate_mode(
        [rng::Random.AbstractRNG,]
        model::DynamicPPL.Model,
        estimator::ModeEstimator,
        solver=OptimizationOptimJL.LBFGS();
        link::Bool=true,
        initial_params=DynamicPPL.InitFromPrior(),
        lb::Union{NamedTuple,AbstractDict{<:VarName,<:Any}}=(;),
        ub::Union{NamedTuple,AbstractDict{<:VarName,<:Any}}=(;),
        adtype::AbstractADType=AutoForwardDiff(),
        check_model::Bool=true,
        check_constraints_at_runtime::Bool=true,
        kwargs...,
    )

Find the mode of the probability distribution of a model.

Under the hood this function constructs a `LogDensityFunction` and calls
`Optimization.solve` on it.

Note that the optimisation interface that Turing exposes is a more high-level interface
which is tailored towards probabilistic modelling, so not every option available in
Optimization.jl is supported here. In particular, Turing's optimisation interface allows you
to:

- Provide initial parameters, lower bounds, and upper bounds as mappings of `VarName`s to
  values in original (unlinked space).

- Choose whether to run the optimisation in linked or unlinked space (by default linked).
  Linked space means that parameters are transformed to unconstrained Euclidean space,
  meaning that you can avoid hard edges in the optimisation landscape (i.e., logpdf
  suddenly dropping to `-Inf` outside the support of a variable). It also avoids cases
  where parameters may not be independent, e.g., `x ~ Dirichlet(...)` where the components
  of `x` must sum to 1. Optimisation in linked space is enabled by default.

Turing is responsible for 'translating' these user-friendly specifications into vectorised
forms (of initial parameters, lower bounds, and upper bounds) that Optimization.jl can work
with.

However, there are cases where this translation can fail or otherwise be ill-defined
(specifically when considering constraints). For example, recall that constraints are
supplied in unlinked space, but the optimisation is run by default in linked space.
Sometimes it is possible to translate constraints from unlinked space to linked space: for
example, for `x ~ Beta(2, 2)`, lower bounds in unlinked space can be translated to lower
bounds in linked space via the logit transform (specificallly, by calling
`to_linked_vec_transform(Beta(2, 2))`.

However, if a user supplies a constraint on a Dirichlet variable, there is no well-defined
mapping of unlinked constraints to linked space. In such cases, Turing will throw an error
(although you can still run in unlinked space). Generic, non-box constraints are also not
possible to correctly support, so Turing's optimisation interface refuses to support them.

See https://github.com/TuringLang/Turing.jl/issues/2634 for more discussion on the interface
and what it supports.

If you need these capabilities, we suggest that you create your own LogDensityFunction and
call Optimization.jl directly on it.

# Arguments

- `rng::Random.AbstractRNG`: an optional random number generator. This is used only for
  parameter initialisation; it does not affect the actual optimisation process.

- `model::DynamicPPL.Model`: The model for which to estimate the mode.

- `estimator::ModeEstimator`: Can be either `MLE()` for maximum likelihood estimation or
  `MAP()` for maximum a posteriori estimation.

- `solver=OptimizationOptimJL.LBFGS()`: The optimization algorithm to use. The default
  solver is L-BFGS, which is a good general-purpose solver that supports box constraints.
  You can also use any solver supported by
  [Optimization.jl](https://docs.sciml.ai/Optimization/stable/). 

# Keyword arguments

- `link::Bool=true`: if true, the model parameters are transformed to an unconstrained
  space for the optimisation. This is generally recommended as it avoids hard edges (i.e.,
  returning a probability of `Inf` outside the support of the parameters), which can lead to
  NaN's or incorrect results. Note that the returned parameter values are always in the
  original (unlinked) space, regardless of whether `link` is true or false.

- `initial_params::DynamicPPL.AbstractInitStrategy=DynamicPPL.InitFromPrior()`: an
  initialisation strategy for the parameters. By default, parameters are initialised by
  generating from the prior. The initialisation strategy will always be augmented by
  any contraints provided via `lb` and `ub`, in that the initial parameters will be
  guaranteed to lie within the provided bounds.

- `lb::Union{NamedTuple,AbstractDict{<:VarName,<:Any}}=(;)`: a mapping from variable names
  to lower bounds for the optimisation. The bounds should be provided in the original
  (unlinked) space. Not all constraints are supported by Turing's optimisation interface.
  See details above.

- `ub::Union{NamedTuple,AbstractDict{<:VarName,<:Any}}=(;)`: a mapping from variable names
  to upper bounds for the optimisation. The bounds should be provided in the original
  (unlinked) space. Not all constraints are supported by Turing's optimisation interface.
  See details above.

- `adtype::AbstractADType=AutoForwardDiff()`: The automatic differentiation backend to use.

- `check_model::Bool=true`: if true, the model is checked for potential errors before
  optimisation begins.

- `check_constraints_at_runtime::Bool=true`: if true, the constraints provided via `lb`
   and `ub` are checked at each evaluation of the log probability during optimisation (even
   though Optimization.jl already has access to these constraints). This can be useful in a
   very specific situation: consider a model where a variable has a dynamic support, e.g.
   `y ~ truncated(Normal(); lower=x)`, where `x` is another variable in the model. In this
   case, if the model is run in linked space, then the box constraints that Optimization.jl
   sees may not always be correct, and `y` may go out of its intended bounds due to changes
   in `x`. Enabling this option will ensure that such violations are caught and an error
   thrown. This is very cheap to do, but if you absolutely need to squeeze out every last
   bit of performance and you know you will not be hitting the edge case above, you can
   disable this check.

Any extra keyword arguments are passed to `Optimization.solve`.
"""
function estimate_mode(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    estimator::ModeEstimator,
    solver=LBFGS();
    link::Bool=true,
    initial_params=DynamicPPL.InitFromPrior(),
    lb::Union{NamedTuple,AbstractDict{<:VarName,<:Any}}=(;),
    ub::Union{NamedTuple,AbstractDict{<:VarName,<:Any}}=(;),
    adtype=ADTypes.AutoForwardDiff(),
    check_model::Bool=true,
    check_constraints_at_runtime::Bool=true,
    solve_kwargs...,
)
    if check_model
        new_model = DynamicPPL.setleafcontext(model, DynamicPPL.InitContext())
        DynamicPPL.check_model(new_model, VarInfo(); error_on_failure=true)
    end

    # Generate a LogDensityFunction first. We do this first because we want to use the
    # info stored in the LDF to generate the initial parameters and constraints in the
    # correct order.
    vi = VarInfo(model)
    vi = if link
        DynamicPPL.link!!(vi, model)
    else
        vi
    end
    getlogdensity = logprob_func(estimator)
    accs = if check_constraints_at_runtime
        (logprob_accs(estimator)..., ConstraintCheckAccumulator(lb, ub))
    else
        logprob_accs(estimator)
    end
    # Note that we don't need adtype to construct the LDF, because it's specified inside the
    # OptimizationProblem.
    ldf = LogDensityFunction(model, getlogdensity, vi, accs)

    # Generate bounds and initial parameters in the unlinked or linked space as requested.
    lb_vec, ub_vec, inits_vec = make_optim_bounds_and_init(
        rng, ldf, Turing._convert_initial_params(initial_params), lb, ub
    )
    # If there are no constraints, then we can omit them from the OptimizationProblem
    # construction. Note that lb and ub must be provided together, not just one of them.
    bounds_kwargs = if any(isfinite, lb_vec) || any(isfinite, ub_vec)
        (lb=lb_vec, ub=ub_vec)
    else
        (;)
    end

    # Insert a negative sign here because Optimization.jl does minimization.
    lp_function = (x, _) -> -LogDensityProblems.logdensity(ldf, x)
    optf = Optimization.OptimizationFunction(lp_function, adtype)
    optprob = Optimization.OptimizationProblem(optf, inits_vec; bounds_kwargs...)
    solution = Optimization.solve(optprob, solver; solve_kwargs...)
    return ModeResult(ldf, solution, link, estimator)
end
function estimate_mode(model::DynamicPPL.Model, args...; kwargs...)
    return estimate_mode(Random.default_rng(), model, args...; kwargs...)
end

"""
    maximum_a_posteriori(
        [rng::Random.AbstractRNG,]
        model::DynamicPPL.Model,
        [solver];
        kwargs...
    )

Find the maximum a posteriori estimate of a model.

This is a convenience function that calls `estimate_mode` with `MAP()` as the estimator.
Please see the documentation of [`Turing.Optimisation.estimate_mode`](@ref) for full
details.
"""
function maximum_a_posteriori(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, args...; kwargs...
)
    return estimate_mode(rng, model, MAP(), args...; kwargs...)
end
function maximum_a_posteriori(model::DynamicPPL.Model, args...; kwargs...)
    return maximum_a_posteriori(Random.default_rng(), model, args...; kwargs...)
end

"""
    maximum_likelihood(
        [rng::Random.AbstractRNG,]
        model::DynamicPPL.Model,
        [solver];
        kwargs...
    )

Find the maximum likelihood estimate of a model.

This is a convenience function that calls `estimate_mode` with `MLE()` as the estimator.
Please see the documentation of [`Turing.Optimisation.estimate_mode`](@ref) for full
details.
"""
function maximum_likelihood(
    rng::Random.AbstractRNG, model::DynamicPPL.Model, args...; kwargs...
)
    return estimate_mode(rng, model, MLE(), args...; kwargs...)
end
function maximum_likelihood(model::DynamicPPL.Model, args...; kwargs...)
    return maximum_likelihood(Random.default_rng(), model, args...; kwargs...)
end

include("stats.jl")

end
