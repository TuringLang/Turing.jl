module Optimisation

using ..Turing
using ..Turing: NamedArrays, ModeResult, MLE, MAP, OptimLogDensity, OptimizationContext, ModeEstimator
using Optimization
using OptimizationOptimJL: LBFGS
using Bijectors
using Random
using SciMLBase: OptimizationFunction, OptimizationProblem, AbstractADType, NoAD, solve, AbstractSciMLAlgorithm

using Accessors: Accessors
using DynamicPPL
using DynamicPPL: Model, VarInfo, istrans

export estimate_mode

#################################################
# Generic optimisation objective initialisation #
#################################################

# TODO Consider rewriting the way transform!! is implemented. This could be done much more
# simply.
function transform!!(f::OptimLogDensity)
    ## Check link status of vi in OptimLogDensity
    linked = DynamicPPL.istrans(f.varinfo)

    ## transform into constrained or unconstrained space depending on current state of vi
    f = Accessors.@set f.varinfo = if !linked
        DynamicPPL.link!!(f.varinfo, f.model)
    else
        DynamicPPL.invlink!!(f.varinfo, f.model)
    end

    return f
end

function transform!!(p::AbstractArray, vi::DynamicPPL.VarInfo, model::DynamicPPL.Model, to_constrained::Val{true})
    !DynamicPPL.istrans(vi) && return p
    vi = DynamicPPL.unflatten(vi, p)
    vi = DynamicPPL.invlink!!(vi, model)
    p .= vi[:]
    DynamicPPL.link!!(vi, model)
    return p
end

# TODO This is currently unused but I think will be used once we start transforming initial
# values.
function transform!!(p::AbstractArray, vi::DynamicPPL.VarInfo, model::DynamicPPL.Model, to_constrained::Val{false})
    linked = DynamicPPL.istrans(vi)
    if linked
        vi = DynamicPPL.invlink!!(vi, model)
    end
    vi = DynamicPPL.unflatten(vi, p)
    vi = DynamicPPL.link!!(vi, model)
    p .= vi[:]

    # If linking mutated, we need to link once more.
    !linked && DynamicPPL.invlink!!(vi, model)

    return p
end

function transform(p::AbstractArray, vi::DynamicPPL.VarInfo, model::DynamicPPL.Model, to_constrained::Bool)
    return transform!!(copy(p), vi, model, Val(to_constrained))
end

function check_bounds(p, ub, lb)
    return all(map(
        (value, upper_bound, lower_bound) -> lower_bound <= value <= upper_bound,
        zip(p, lb, ub)
    ))
end

"Check that given parameters `p` respect the bounds and constraints."
function isinbounds(p, ub, lb, cons, lcons, ucons)
    if ub !== nothing && !(all(p .<= ub))
        return false
    end
    if lb !== nothing && !(all(p .>= lb))
        return false
    end
    if cons !== nothing
        c = cons(p)
        if ucons !== nothing && !(c .<= ucons)
            return false
        end
        if lcons !== nothing && !(c .>= lcons)
            return false
        end
    end
    return true
end

const MAX_INIT_SAMPLES = 100

function init(model::DynamicPPL.Model; ub=nothing, lb=nothing, cons=nothing, lcons=nothing, ucons=nothing)
    # TODO Switch to sampling uniformly within the bounds when box constraints are present.
    init_value = collect(values(rand(model)))
    counter = 1
    while !isinbounds(init_value, ub, lb, cons, lcons, ucons) && counter < MAX_INIT_SAMPLES
        init_value = collect(values(rand(model)))
        counter += 1
    end
    if counter == MAX_BOUNDS_ATTEMPTS
        throw(ArgumentError("Could not find a valid initial value by sampling from the prior. You may need to provide an initial value."))
    end
    return init_value
end

function optim_objective(model::DynamicPPL.Model, estimator::ModeEstimator)
    inner_context = estimator isa MAP ? DefaultContext() : LikelihoodContext()
    ctx = OptimizationContext(inner_context)
    obj = OptimLogDensity(model, ctx)
    return obj
end

function estimate_mode(model::DynamicPPL.Model, estimator::ModeEstimator, init_value::Union{AbstractVector,Nothing}, args...; kwargs...)
    return estimate_mode(model, estimator, init_value, LBFGS(), args...; kwargs...)
end

function estimate_mode(model::DynamicPPL.Model, estimator::ModeEstimator, solver, args...; kwargs...)
    return estimate_mode(model, estimator, nothing, solver, args...; kwargs...)
end

function estimate_mode(model::DynamicPPL.Model, estimator::ModeEstimator, args...; kwargs...)
    return estimate_mode(model, estimator, nothing, LBFGS(), args...; kwargs...)
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
    hasconstraints = ub !== nothing || lb !== nothing || cons !== nothing || lcons !== nothing || ucons !== nothing

    # Create the objective function for the optimization. It is nothing other than the
    # negative log density of the model, either for the likelihood or the posterior.
    obj = optim_objective(model, estimator)
    l(x, _) = obj(x)
    f = OptimizationFunction(l, adtype; cons=cons)

    if init_value === nothing
        init_value = init(model; ub=ub, lb=lb, lcons=lcons, ucons=ucons)
    end

    # TODO We currently couple together the questions of whether the user specified
    # bounds/constraints, and whether we transform the objective function to an
    # unconstrained space. These should be separate concerns, but for that we need to
    # implement getting the bounds of the prior distributions.
    if !hasconstraints
        # If the user hasn't specified any constraints, transform the objective function and
        # its initial value to the unconstrained space and run the optimization there.
        obj = transform!!(obj)
        # TODO Isn't this transform the wrong way around?
        init_value = transform(init_value, obj.varinfo, model, true)
        prob = OptimizationProblem(f, init_value)
    else
        # If the user has specified constraints, run a constrained optimization in the
        # untransformed space.
        prob = OptimizationProblem(f, init_value; lcons=lcons, ucons=ucons, lb=lb, ub=ub)
    end

    solution = solve(prob, solver, args...; kwargs...)
    solution_values = solution.u
    if !hasconstraints
        # If we ran the optimization in the unconstrained space, transform the solution back
        # to the original space.
        solution_values = transform(solution_values, obj.varinfo, model, true)
    end

    # Make one transition to get the parameter names.
    ts = [Turing.Inference.Transition(
        Turing.Inference.getparams(model, obj.varinfo),
        DynamicPPL.getlogp(obj.varinfo)
    )]
    varnames = map(Symbol, first(Turing.Inference._params_to_array(model, ts)))

    # Store the parameters and their names in an array.
    vmat = NamedArrays.NamedArray(solution_values, varnames)

    return ModeResult(vmat, solution, -solution.minimum, obj)
end

function maximum_a_posteriori(model::DynamicPPL.Model, args...; kwargs...)
    return estimate_mode(model, MAP(), args...; kwargs...)
end

function maximum_likelihood(model::DynamicPPL.Model, args...; kwargs...)
    return estimate_mode(model, MLE(), args...; kwargs...)
end

end
