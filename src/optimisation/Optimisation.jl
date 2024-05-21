module Optimisation

using ..Turing
using NamedArrays: NamedArrays
using DynamicPPL: DynamicPPL
import LogDensityProblems: LogDensityProblems
import LogDensityProblemsAD: LogDensityProblemsAD
using Optimization: Optimization
using OptimizationOptimJL: OptimizationOptimJL
using DocStringExtensions: DocStringExtensions
using Bijectors: Bijectors
using Random: Random
using SciMLBase: SciMLBase
using ADTypes: ADTypes
using StatsBase: StatsBase
using Accessors: Accessors
using Printf: Printf
using ForwardDiff: ForwardDiff
using StatsAPI: StatsAPI
using Statistics: Statistics

export estimate_mode, maximum_a_posteriori, maximum_likelihood, MLE, MAP, ModeResult,
    OptimLogDensity, OptimizationContext

"""
    ModeEstimator

An abstract type to mark whether mode estimation is to be done with maximum a posteriori
(MAP) or maximum likelihood estimation (MLE).
"""
abstract type ModeEstimator end
struct MLE <: ModeEstimator end
struct MAP <: ModeEstimator end

"""
    OptimizationContext{C<:AbstractContext} <: AbstractContext

The `OptimizationContext` transforms variables to their constrained space, but
does not use the density with respect to the transformation. This context is
intended to allow an optimizer to sample in R^n freely.
"""
struct OptimizationContext{C<:DynamicPPL.AbstractContext} <: DynamicPPL.AbstractContext
    context::C

    function OptimizationContext{C}(context::C) where {C<:DynamicPPL.AbstractContext}
        if !(context isa Union{DynamicPPL.DefaultContext,DynamicPPL.LikelihoodContext})
            throw(ArgumentError(
                "`OptimizationContext` supports only leaf contexts of type "
                * "`DynamicPPL.DefaultContext` and `DynamicPPL.LikelihoodContext` "
                * "(given: `$(typeof(context)))`"
            ))
        end
        return new{C}(context)
    end
end

OptimizationContext(ctx::DynamicPPL.AbstractContext) = OptimizationContext{typeof(ctx)}(ctx)

DynamicPPL.NodeTrait(::OptimizationContext) = DynamicPPL.IsLeaf()

function DynamicPPL.tilde_assume(ctx::OptimizationContext, dist, vn, vi)
    r = vi[vn, dist]
    lp = if ctx.context isa DynamicPPL.DefaultContext
        # MAP
        Distributions.logpdf(dist, r)
    else
        # MLE
        0
    end
    return r, lp, vi
end

_loglikelihood(dist::Distribution, x) = StatsAPI.loglikelihood(dist, x)

function _loglikelihood(dists::AbstractArray{<:Distribution}, x)
    return StatsAPI.loglikelihood(arraydist(dists), x)
end

function DynamicPPL.dot_tilde_assume(ctx::OptimizationContext, right, left, vns, vi)
    # Values should be set and we're using `SampleFromPrior`, hence the `rng` argument
    # shouldn't affect anything.
    # TODO: Stop using `get_and_set_val!`.
    r = DynamicPPL.get_and_set_val!(
        Random.default_rng(), vi, vns, right, DynamicPPL.SampleFromPrior()
    )
    lp = if ctx.context isa DynamicPPL.DefaultContext
        # MAP
        _loglikelihood(right, r)
    else
        # MLE
        0
    end
    return r, lp, vi
end

"""
    OptimLogDensity{M<:DynamicPPL.Model,C<:Context,V<:DynamicPPL.VarInfo}

A struct that stores the negative log density function of a `DynamicPPL` model.
"""
const OptimLogDensity{M<:DynamicPPL.Model,C<:OptimizationContext,V<:DynamicPPL.VarInfo} = Turing.LogDensityFunction{V,M,C}

"""
    OptimLogDensity(model::DynamicPPL.Model, context::OptimizationContext)

Create a callable `OptimLogDensity` struct that evaluates a model using the given `context`.
"""
function OptimLogDensity(model::DynamicPPL.Model, context::OptimizationContext)
    init = DynamicPPL.VarInfo(model)
    return Turing.LogDensityFunction(init, model, context)
end

"""
    (f::OptimLogDensity)(z)

Evaluate the negative log joint or log likelihood at the array `z`. Which one is evaluated
depends on the context of `f`.
"""
function (f::OptimLogDensity)(z::AbstractVector)
    varinfo = DynamicPPL.unflatten(f.varinfo, z)
    return -DynamicPPL.getlogp(last(DynamicPPL.evaluate!!(f.model, varinfo, f.context)))
end

# NOTE: This seems a bit weird IMO since this is the _negative_ log-likelihood.
LogDensityProblems.logdensity(f::OptimLogDensity, z::AbstractVector) = f(z)

# NOTE: The format of this function is dictated by Optim. The first argument sets whether to
# compute the function value, the second whether to compute the gradient (and stores the
# gradient). The last one is the actual argument of the objective function.
function (f::OptimLogDensity)(F, G, z)
    if G !== nothing
        # Calculate negative log joint and its gradient.
        # TODO: Make OptimLogDensity already an LogDensityProblems.ADgradient? Allow to specify AD?
        ℓ = LogDensityProblemsAD.ADgradient(f)
        neglogp, ∇neglogp = LogDensityProblems.logdensity_and_gradient(ℓ, z)

        # Save the gradient to the pre-allocated array.
        copyto!(G, ∇neglogp)

        # If F is something, the negative log joint is requested as well.
        # We have already computed it as a by-product above and hence return it directly.
        if F !== nothing
            return neglogp
        end
    end

    # Only negative log joint requested but no gradient.
    if F !== nothing
        return LogDensityProblems.logdensity(f, z)
    end

    return nothing
end

"""
    ModeResult{
        V<:NamedArrays.NamedArray,
        M<:NamedArrays.NamedArray,
        O<:Optim.MultivariateOptimizationResults,
        S<:NamedArrays.NamedArray
    }

A wrapper struct to store various results from a MAP or MLE estimation.
"""
struct ModeResult{
    V<:NamedArrays.NamedArray,
    O<:Any,
    M<:OptimLogDensity
} <: StatsBase.StatisticalModel
    "A vector with the resulting point estimates."
    values::V
    "The stored optimiser results."
    optim_result::O
    "The final log likelihood or log joint, depending on whether `MAP` or `MLE` was run."
    lp::Float64
    "The evaluation function used to calculate the output."
    f::M
end

#############################
# Various StatsBase methods #
#############################

function Base.show(io::IO, ::MIME"text/plain", m::ModeResult)
    print(io, "ModeResult with maximized lp of ")
    Printf.@printf(io, "%.2f", m.lp)
    println(io)
    show(io, m.values)
end

function Base.show(io::IO, m::ModeResult)
    show(io, m.values.array)
end

function StatsBase.coeftable(m::ModeResult; level::Real=0.95)
    # Get columns for coeftable.
    terms = string.(StatsBase.coefnames(m))
    estimates = m.values.array[:, 1]
    stderrors = StatsBase.stderror(m)
    zscore = estimates ./ stderrors
    p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:both), zscore)

    # Confidence interval (CI)
    q = Statistics.quantile(Distributions.Normal(), (1 + level) / 2)
    ci_low = estimates .- q .* stderrors
    ci_high = estimates .+ q .* stderrors

    level_ = 100 * level
    level_percentage = isinteger(level_) ? Int(level_) : level_

    StatsBase.CoefTable(
        [estimates, stderrors, zscore, p, ci_low, ci_high],
        ["Coef.", "Std. Error", "z", "Pr(>|z|)", "Lower $(level_percentage)%", "Upper $(level_percentage)%"],
        terms)
end

function StatsBase.informationmatrix(m::ModeResult; hessian_function=ForwardDiff.hessian, kwargs...)
    # Calculate Hessian and information matrix.

    # Convert the values to their unconstrained states to make sure the
    # Hessian is computed with respect to the untransformed parameters.
    linked = DynamicPPL.istrans(m.f.varinfo)
    if linked
        m = Accessors.@set m.f.varinfo = DynamicPPL.invlink!!(m.f.varinfo, m.f.model)
    end

    # Calculate the Hessian, which is the information matrix because the negative of the log likelihood was optimized
    varnames = StatsBase.coefnames(m)
    info = hessian_function(m.f, m.values.array[:, 1])

    # Link it back if we invlinked it.
    if linked
        m = Accessors.@set m.f.varinfo = DynamicPPL.link!!(m.f.varinfo, m.f.model)
    end

    return NamedArrays.NamedArray(info, (varnames, varnames))
end

StatsBase.coef(m::ModeResult) = m.values
StatsBase.coefnames(m::ModeResult) = names(m.values)[1]
StatsBase.params(m::ModeResult) = StatsBase.coefnames(m)
StatsBase.vcov(m::ModeResult) = inv(StatsBase.informationmatrix(m))
StatsBase.loglikelihood(m::ModeResult) = m.lp

"""
    ModeEstimationConstraints

A struct that holds constraints for mode estimation problems.

The fields are the same as possible constraints supported by the Optimization.jl:
`ub` and `lb` specify lower and upper bounds of box constraints. `cons` is a function that
takes the parameters of the model and returns a list of derived quantities, which are then
constrained by the lower and upper bounds set by `lcons` and `ucons`. We refer to these
as generic constraints. Please see the documentation of
[Optimization.jl](https://docs.sciml.ai/Optimization/stable/) for more details.

Any of the fields can be `nothing`, disabling the corresponding constraints.
"""
struct ModeEstimationConstraints{
    Lb<:Union{Nothing,AbstractVector},
    Ub<:Union{Nothing,AbstractVector},
    Cons,
    LCons<:Union{Nothing,AbstractVector},
    UCons<:Union{Nothing,AbstractVector},
}
    lb::Lb
    ub::Ub
    cons::Cons
    lcons::LCons
    ucons::UCons
end

has_box_constraints(c::ModeEstimationConstraints) = c.ub !== nothing || c.lb !== nothing
has_generic_constraints(c::ModeEstimationConstraints) = (
    c.cons !== nothing || c.lcons !== nothing || c.ucons !== nothing
)
has_constraints(c) = has_box_constraints(c) || has_generic_constraints(c)

"""
    generate_initial_params(model::DynamicPPL.Model, initial_params, constraints)

Generate an initial value for the optimization problem.

If `initial_params` is not `nothing`, a copy of it is returned. Otherwise initial parameter
values are generated either by sampling from the prior (if no constraints are present) or
uniformly from the box constraints. If generic constraints are set, an error is thrown.
"""
function generate_initial_params(model::DynamicPPL.Model, initial_params, constraints)
    if initial_params !== nothing
        return copy(initial_params)
    end
    if has_generic_constraints(constraints)
        throw(ArgumentError(
            "You must provide an initial value when using generic constraints."
        ))
    end
    if has_box_constraints(constraints)
        return [
            rand(Distributions.Uniform(lower, upper))
            for (lower, upper) in zip(constraints.lb, constraints.ub)
        ]
    end
    return rand(Vector, model)
end

"""
    ModeEstimationProblem{M,E,Iv,Cons,OLD,ADType}

A struct that defines a mode estimation problem.

# Fields
$(DocStringExtensions.TYPEDFIELDS)
"""
struct ModeEstimationProblem{
    M<:DynamicPPL.Model,
    E<:ModeEstimator,
    Iv<:AbstractVector,
    Cons<:ModeEstimationConstraints,
    OLD<:OptimLogDensity,
    ADType<:ADTypes.AbstractADType,
}
    "Model for which to estimate the mode."
    model::M
    """Mode estimator. Either [`MLE()`](@ref) for maximum likelihood or [`MAP()`](@ref) for
    maximum a posteriori."""
    estimator::E
    "Initial value for the optimization."
    initial_params::Iv
    "Constraints for the optimization problem."
    constraints::Cons
    "Objective function for the optimization."
    log_density::OLD
    "Automatic differentiation type to use. See ADTypes.jl."
    adtype::ADType
    "Whether the objective function is transformed to an unconstrained space."
    linked::Bool
end

"""
    ModeEstimationProblem(args...)

Create a mode estimation problem.

# Arguments:
- `model::DynamicPPL.Model`: The model for which to estimate the mode.
- `estimator::ModeEstimator`: Can be either `MLE()` for maximum likelihood estimation or
`MAP()` for maximum a posteriori estimation.
- `initial_params::Union{AbstractVector,Nothing}`: Initial value for the optimization.
If `nothing`, a default value is generated by [`generate_initial_params`](@ref).
- `lb::Union{Nothing,AbstractVector}`: See [`ModeEstimationConstraints`](@ref).
- `ub::Union{Nothing,AbstractVector}`: As above.
- `cons::Union{Nothing,Function}`: As above.
- `lcons::Union{Nothing,AbstractVector}`: As above.
- `ucons::Union{Nothing,AbstractVector}`: As above.
- `adtype::AbstractADType=AutoForwardDiff()`: The automatic differentiation type to use.

The problem is always created with `linked=false`. See [`link`](@ref) for how to transform
it.
"""
function ModeEstimationProblem(
    model, estimator, initial_params, lb, ub, cons, lcons, ucons, adtype
)
    constraints = ModeEstimationConstraints(lb, ub, cons, lcons, ucons)
    initial_params = generate_initial_params(model, initial_params, constraints)
    inner_context = if estimator isa MAP
        DynamicPPL.DefaultContext()
    else
        DynamicPPL.LikelihoodContext()
    end
    ctx = OptimizationContext(inner_context)
    log_density = OptimLogDensity(model, ctx)
    return ModeEstimationProblem(
        model, estimator, initial_params, constraints, log_density, adtype, false
    )
end

has_box_constraints(p::ModeEstimationProblem) = has_box_constraints(p.constraints)
has_generic_constraints(p::ModeEstimationProblem) = has_generic_constraints(p.constraints)

function default_solver(problem::ModeEstimationProblem)
    return if has_generic_constraints(problem.constraints)
        OptimizationOptimJL.IPNewton()
    else
        OptimizationOptimJL.LBFGS()
    end
end

"""
    link(p::ModeEstimationProblem)

Transform `p` to unconstrained space, where all parameters can take any real value.
"""
function link(p::ModeEstimationProblem)
    if has_constraints(p)
        throw(
            "Transforming constrained optimisation problems to unconstrained space is " *
            "not yet implemented."
        )
    end
    # Note that redefining obj and initial_params out of place is intentional: It avoids
    # issues with models for which linking changes the parameter space dimension.
    ld = p.log_density
    varinfo = DynamicPPL.link(
        DynamicPPL.unflatten(ld.varinfo, copy(p.initial_params)), ld.model
    )
    ld = Accessors.@set ld.varinfo = varinfo
    initial_params = varinfo[:]
    return ModeEstimationProblem(
        p.model, p.estimator, initial_params, p.constraints, ld, p.adtype, true
    )
end

"""
    OptimizationProblem(prob::ModeEstimationProblem)

Create a SciML `OptimizationProblem` from `prob`.
"""
function Optimization.OptimizationProblem(prob::ModeEstimationProblem)
    c = prob.constraints
    # Note that OptimLogDensity is a callable that evaluates the model with given
    # parameters. Hence we can use it as the objective function.
    f = Optimization.OptimizationFunction((x, _) -> prob.log_density(x), prob.adtype; cons=c.cons)
    opt_prob = if !has_constraints(prob)
        Optimization.OptimizationProblem(f, prob.initial_params)
    else
        Optimization.OptimizationProblem(
            f, prob.initial_params;
            lcons=c.lcons, ucons=c.ucons, lb=c.lb, ub=c.ub
        )
    end
    return opt_prob
end

"""
    ModeResult(prob::ModeEstimationProblem, solution::AbstractVector)

Create a `ModeResult` for a given estimation problem and a solution given by `solve`.

`Optimization.solve` returns its own result type. This function converts that into the
richer format of `ModeResult`. It also takes care of transforming them back to the original
parameter space in case the optimization was done in a transformed space.
"""
function ModeResult(prob::ModeEstimationProblem, solution::AbstractVector)
    varinfo_new = DynamicPPL.unflatten(prob.log_density.varinfo, solution.u)
    # `getparams` performs invlinking if needed
    vns_vals_iter = Turing.Inference.getparams(prob.model, varinfo_new)
    syms = map(Symbol ∘ first, vns_vals_iter)
    vals = map(last, vns_vals_iter)
    return ModeResult(
        NamedArrays.NamedArray(vals, syms), solution, -solution.objective, prob.log_density
    )
end

"""
    estimate_mode(
        model::DynamicPPL.Model,
        estimator::ModeEstimator,
        [initial_params::Union{AbstractVector,Nothing}],
        [solver];
        <keyword arguments>,
    )

Find the mode of the probability distribution of a model.

Under the hood this function calls `Optimization.solve`.

# Arguments
- `model::DynamicPPL.Model`: The model for which to estimate the mode.
- `estimator::ModeEstimator`: Can be either `MLE()` for maximum likelihood estimation or
`MAP()` for maximum a posteriori estimation.
- `initial_params::Union{AbstractVector,Nothing}`: Initial value for the optimization.
Optional. If `nothing` or omitted it is generated by either sampling from the prior
distribution or uniformly from the box constraints, if any.
- `solver`. The optimization algorithm to use. Optional. Can be any solver recognised by
Optimization.jl. If `nothing` or omitted a default solver is used: LBFGS, or IPNewton if
non-box constraints are present.

# Keyword arguments
- `adtype::AbstractADType=AutoForwardDiff()`: The automatic differentiation type to use.
- Keyword arguments `lb`, `ub`, `cons`, `lcons`, and `ucons` define constraints for the
optimization problem. Please see the documentation of [`ModeEstimationConstraints`](@ref)
for more details.
- Any extra keyword arguments are passed to `Optimization.solve`.
"""
function estimate_mode(
    model::DynamicPPL.Model,
    estimator::ModeEstimator,
    initial_params::Union{AbstractVector,Nothing},
    solver;
    adtype=ADTypes.AutoForwardDiff(),
    cons=nothing,
    lcons=nothing,
    ucons=nothing,
    lb=nothing,
    ub=nothing,
    kwargs...
)
    prob = ModeEstimationProblem(
        model, estimator, initial_params, lb, ub, cons, lcons, ucons, adtype
    )
    if solver === nothing
        solver = default_solver(prob)
    end
    # TODO(mhauru) We currently couple together the questions of whether the user specified
    # bounds/constraints and whether we transform the objective function to an
    # unconstrained space. These should be separate concerns, but for that we need to
    # implement getting the bounds of the prior distributions.
    optimise_in_unconstrained_space = !has_constraints(prob)
    if optimise_in_unconstrained_space
        prob = link(prob)
    end
    solution = Optimization.solve(Optimization.OptimizationProblem(prob), solver; kwargs...)
    # TODO(mhauru) We return a ModeResult for compatibility with the older Optim.jl
    # interface. Might we want to break that and develop a better return type?
    return ModeResult(prob, solution)
end

# If no solver is given.
function estimate_mode(
    model::DynamicPPL.Model,
    estimator::ModeEstimator,
    initial_params::Union{AbstractVector,Nothing},
    args...;
    kwargs...
)
    return estimate_mode(model, estimator, initial_params, nothing, args...; kwargs...)
end

# If no initial value is given.
function estimate_mode(
    model::DynamicPPL.Model,
    estimator::ModeEstimator,
    solver,
    args...;
    kwargs...
)
    return estimate_mode(model, estimator, nothing, solver, args...; kwargs...)
end

# If no solver or initial value is given.
function estimate_mode(
    model::DynamicPPL.Model,
    estimator::ModeEstimator,
    args...;
    kwargs...
)
    return estimate_mode(model, estimator, nothing, nothing, args...; kwargs...)
end

"""
    maximum_a_posteriori(
        model::DynamicPPL.Model,
        [initial_params::Union{AbstractVector,Nothing}],
        [solver];
        <keyword arguments>
        )

Find the maximum a posteriori estimate of a model.

This is a convenience function that calls `estimate_mode` with `MAP()` as the estimator.
Please see the documentation of [`estimate_mode`](@ref) for more details.
"""
function maximum_a_posteriori(model::DynamicPPL.Model, args...; kwargs...)
    return estimate_mode(model, MAP(), args...; kwargs...)
end

"""
    maximum_likelihood(
        model::DynamicPPL.Model,
        [initial_params::Union{AbstractVector,Nothing}],
        [solver];
        <keyword arguments>
        )

Find the maximum likelihood estimate of a model.

This is a convenience function that calls `estimate_mode` with `MLE()` as the estimator.
Please see the documentation of [`estimate_mode`](@ref) for more details.
"""
function maximum_likelihood(model::DynamicPPL.Model, args...; kwargs...)
    return estimate_mode(model, MLE(), args...; kwargs...)
end

end
