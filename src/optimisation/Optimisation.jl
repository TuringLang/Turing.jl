module Optimisation

using ..Turing
using ..Turing: NamedArrays, ModeResult, MLE, MAP, OptimLogDensity, OptimizationContext, ModeEstimator
using Optimization
using OptimizationOptimJL: LBFGS, IPNewton
using Bijectors
using Random
using SciMLBase: OptimizationFunction, OptimizationProblem, AbstractADType, NoAD, solve, AbstractSciMLAlgorithm

using Accessors: Accessors
using DynamicPPL
using DynamicPPL: Model, VarInfo, istrans

export estimate_mode, maximum_a_posteriori, maximum_likelihood

function ensure_init_value(model::DynamicPPL.Model, init_value, constraints)
    if init_value !== nothing
        return copy(init_value)
    end
    if has_generic_constraints(constraints)
        throw(ArgumentError("You must provide an initial value when using generic constraints."))
    end
    if has_box_constraints(constraints)
        return [rand(Uniform(lower, upper)) for (lower, upper) in zip(constraints.lb, constraints.ub)]
    end
    return collect(Iterators.flatten(values(rand(model))))
end

struct ModeEstimationConstraints{
    Lb<:Union{Nothing,AbstractVector},
    Ub<:Union{Nothing,AbstractVector},
    Conns,
    LConns<:Union{Nothing,AbstractVector},
    UConns<:Union{Nothing,AbstractVector},
}
    lb::Lb
    ub::Ub
    cons::Conns
    lcons::LConns
    ucons::UConns
end

has_box_constraints(c::ModeEstimationConstraints) = c.ub !== nothing || c.lb !== nothing
has_generic_constraints(c::ModeEstimationConstraints) = c.cons !== nothing || c.lcons !== nothing || c.ucons !== nothing
has_constraints(c::ModeEstimationConstraints) = has_box_constraints(c) || has_generic_constraints(c)

# TODO Do I want this type, or some type like this? I think I want some type that holds both
# the ModeLogDensity and the initial values, so that I can link/invlink them with a single
# function call. Other than that, very unsure what the right abstraction here is.
struct ModeEstimationProblem{
    M<:DynamicPPL.Model,
    E<:ModeEstimator,
    Iv<:AbstractVector,
    Cons<:ModeEstimationConstraints,
    OLD<:OptimLogDensity,
    ADType<:AbstractADType,
}
    model::M
    estimator::E
    init_value::Iv
    constraints::Cons
    log_density::OLD
    adtype::ADType
    linked::Bool
end

function ModeEstimationProblem(model, estimator, init_value, lb, ub, cons, lcons, ucons, adtype)
    constraints = ModeEstimationConstraints(lb, ub, cons, lcons, ucons)
    init_value = ensure_init_value(model, init_value, constraints)
    inner_context = estimator isa MAP ? DefaultContext() : LikelihoodContext()
    ctx = OptimizationContext(inner_context)
    log_density = OptimLogDensity(model, ctx)
    return ModeEstimationProblem(model, estimator, init_value, constraints, log_density, adtype, false)
end

has_box_constraints(p::ModeEstimationProblem) = has_box_constraints(p.constraints)
has_generic_constraints(p::ModeEstimationProblem) = has_generic_constraints(p.constraints)
has_constraints(p::ModeEstimationProblem) = has_constraints(p.constraints)

function default_solver(problem::ModeEstimationProblem)
    return has_generic_constraints(problem.constraints) ? IPNewton() : LBFGS()
end

function link(p::ModeEstimationProblem)
    ld = p.log_density
    ld = Accessors.@set ld.varinfo = DynamicPPL.unflatten(ld.varinfo, copy(p.init_value))
    ld = Accessors.@set ld.varinfo = DynamicPPL.link(ld.varinfo, ld.model)
    init_value = ld.varinfo[:]
    return ModeEstimationProblem(p.model, p.estimator, init_value, p.constraints, ld, p.adtype, true)
end

function OptimizationProblem(me_prob::ModeEstimationProblem)
    # Create the objective function for the optimization. It is nothing other than the
    # negative log density of the model, either for the likelihood or the posterior.
    c = me_prob.constraints
    l(x, _) = me_prob.log_density(x)
    f = OptimizationFunction(l, me_prob.adtype; cons=c.cons)
    if !has_constraints(me_prob)
        opt_prob = OptimizationProblem(f, me_prob.init_value)
    else
        # If the user has specified constraints, run a constrained optimization in the
        # untransformed space.
        opt_prob = OptimizationProblem(f, me_prob.init_value; lcons=c.lcons, ucons=c.ucons, lb=c.lb, ub=c.ub)
    end
    return opt_prob
end

function variable_names(lg::OptimLogDensity)
    return map(Symbol ∘ first, Turing.Inference.getparams(lg.model, lg.varinfo))
end

function ModeResult(prob::ModeEstimationProblem, solution)
    solution_values = solution.u
    ld = prob.log_density
    if prob.linked
        ld = Accessors.@set ld.varinfo = DynamicPPL.unflatten(ld.varinfo, solution_values)
        ld = Accessors.@set ld.varinfo = DynamicPPL.invlink(ld.varinfo, ld.model)
        solution_values = ld.varinfo[:]
    end
    # Store the parameters and their names in an array.
    varnames = variable_names(prob.log_density)
    vmat = NamedArrays.NamedArray(solution_values, varnames)
    return ModeResult(vmat, solution, -solution.minimum, prob.log_density)
end

function estimate_mode(model::DynamicPPL.Model, estimator::ModeEstimator, init_value::Union{AbstractVector,Nothing}, args...; kwargs...)
    return estimate_mode(model, estimator, init_value, nothing, args...; kwargs...)
end

function estimate_mode(model::DynamicPPL.Model, estimator::ModeEstimator, solver, args...; kwargs...)
    return estimate_mode(model, estimator, nothing, solver, args...; kwargs...)
end

function estimate_mode(model::DynamicPPL.Model, estimator::ModeEstimator, args...; kwargs...)
    return estimate_mode(model, estimator, nothing, nothing, args...; kwargs...)
end

function estimate_mode(
    model::DynamicPPL.Model,
    estimator::ModeEstimator,
    init_value::Union{AbstractVector,Nothing},
    solver,
    args...;
    adtype=AutoForwardDiff(),
    cons=nothing,
    lcons=nothing,
    ucons=nothing,
    lb=nothing,
    ub=nothing,
    kwargs...
)
    prob = ModeEstimationProblem(model, estimator, init_value, lb, ub, cons, lcons, ucons, adtype)

    solver = (solver === nothing) ? default_solver(prob) : solver

    # TODO(mhauru) We currently couple together the questions of whether the user specified
    # bounds/constraints, and whether we transform the objective function to an
    # unconstrained space. These should be separate concerns, but for that we need to
    # implement getting the bounds of the prior distributions.
    optimise_in_unconstrained_space = !has_constraints(prob)
    if optimise_in_unconstrained_space
        # If the user hasn't specified any constraints, transform the objective function and
        # its initial value to the unconstrained space and run the optimization there.
        # Note that not mutating obj and init_value in place is intentional: It avoids
        # issues with models for which linking changes the parameter space dimension.
        prob = link(prob)
    end

    solution = solve(OptimizationProblem(prob), solver, args...; kwargs...)
    result = ModeResult(prob, solution)
    return result
end

function maximum_a_posteriori(model::DynamicPPL.Model, args...; kwargs...)
    return estimate_mode(model, MAP(), args...; kwargs...)
end

function maximum_likelihood(model::DynamicPPL.Model, args...; kwargs...)
    return estimate_mode(model, MLE(), args...; kwargs...)
end

end
