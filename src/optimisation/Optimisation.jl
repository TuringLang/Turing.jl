module Optimisation

using ..Turing
using NamedArrays: NamedArrays
using AbstractPPL: AbstractPPL
using DynamicPPL: DynamicPPL
using DocStringExtensions: TYPEDFIELDS
using LogDensityProblems: LogDensityProblems
using Optimization: Optimization
using OptimizationOptimJL: OptimizationOptimJL
using Random: Random
using SciMLBase: SciMLBase
using ADTypes: ADTypes
using StatsBase: StatsBase
using Accessors: Accessors
using Printf: Printf
using ForwardDiff: ForwardDiff
using StatsAPI: StatsAPI
using Statistics: Statistics
using LinearAlgebra: LinearAlgebra

export maximum_a_posteriori, maximum_likelihood, MAP, MLE

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

"""
    MAP <: ModeEstimator

Concrete type for maximum a posteriori estimation.
"""
struct MAP <: ModeEstimator end

"""
    OptimLogDensity{L<:DynamicPPL.LogDensityFunction}

A struct that represents a log-density function, which can be used with Optimization.jl.
This is a thin wrapper around `DynamicPPL.LogDensityFunction`: the main difference is that
the log-density is negated (because Optimization.jl performs minimisation, and we usually
want to maximise the log-density).

An `OptimLogDensity` does not, in itself, obey the LogDensityProblems.jl interface. Thus, if
you want to calculate the log density of its contents at the point `z`, you should manually
call `LogDensityProblems.logdensity(f.ldf, z)`, instead of `LogDensityProblems.logdensity(f,
z)`.

However, because Optimization.jl requires the objective function to be callable, you can
also call `f(z)` directly to get the negative log density at `z`.
"""
struct OptimLogDensity{L<:DynamicPPL.LogDensityFunction}
    ldf::L
end

"""
    (f::OptimLogDensity)(z)
    (f::OptimLogDensity)(z, _)

Evaluate the negative log probability density at the array `z`. Which kind of probability
density is evaluated depends on the `getlogdensity` function used to construct the
underlying `LogDensityFunction` (e.g., `DynamicPPL.getlogjoint` for MAP estimation, or
`DynamicPPL.getloglikelihood` for MLE).

Any second argument is ignored. The two-argument method only exists to match the interface
required by Optimization.jl.
"""
(f::OptimLogDensity)(z::AbstractVector) = -LogDensityProblems.logdensity(f.ldf, z)
(f::OptimLogDensity)(z, _) = f(z)

"""
    ModeResult{
        V<:NamedArrays.NamedArray,
        O<:Any,
        M<:OptimLogDensity,
        P<:AbstractDict{<:VarName,<:Any}
        E<:ModeEstimator,
    }

A wrapper struct to store various results from a MAP or MLE estimation.

## Fields

$(TYPEDFIELDS)
"""
struct ModeResult{
    V<:NamedArrays.NamedArray,
    O<:Any,
    M<:OptimLogDensity,
    P<:AbstractDict{<:AbstractPPL.VarName,<:Any},
    E<:ModeEstimator,
} <: StatsBase.StatisticalModel
    "A vector with the resulting point estimates."
    values::V
    "The stored optimiser results."
    optim_result::O
    "The final log likelihood or log joint, depending on whether `MAP` or `MLE` was run."
    lp::Float64
    "The evaluation function used to calculate the output."
    f::M
    "Dictionary of parameter values"
    params::P
    "Whether the optimization was done in a transformed space."
    linked::Bool
    "The type of mode estimation (MAP or MLE)."
    estimator::E
end

function Base.show(io::IO, ::MIME"text/plain", m::ModeResult)
    print(io, "ModeResult with maximized lp of ")
    Printf.@printf(io, "%.2f", m.lp)
    println(io)
    return show(io, m.values)
end

function Base.show(io::IO, m::ModeResult)
    return show(io, m.values.array)
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

"""
    Base.get(m::ModeResult, var_symbol::Symbol)
    Base.get(m::ModeResult, var_symbols::AbstractVector{Symbol})

Return the values of all the variables with the symbol(s) `var_symbol` in the mode result
`m`. The return value is a `NamedTuple` with `var_symbols` as the key(s). The second
argument should be either a `Symbol` or a vector of `Symbol`s.
"""
function Base.get(m::ModeResult, var_symbols::AbstractVector{Symbol})
    vals_dict = m.params
    iters = map(AbstractPPL.varname_and_value_leaves, keys(vals_dict), values(vals_dict))
    vns_and_vals = mapreduce(collect, vcat, iters)
    varnames = collect(map(first, vns_and_vals))
    # For each symbol s in var_symbols, pick all the values from m.values for which the
    # variable name has that symbol.
    et = eltype(m.values)
    value_vectors = Vector{et}[]
    for s in var_symbols
        push!(
            value_vectors,
            [m.values[Symbol(vn)] for vn in varnames if DynamicPPL.getsym(vn) == s],
        )
    end
    return (; zip(var_symbols, value_vectors)...)
end

Base.get(m::ModeResult, var_symbol::Symbol) = get(m, [var_symbol])

"""
    ModeResult(
        log_density::OptimLogDensity,
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
    log_density::OptimLogDensity,
    solution::SciMLBase.OptimizationSolution,
    linked::Bool,
    estimator::ModeEstimator,
)
    vals = DynamicPPL.ParamsWithStats(solution.u, log_density.ldf).params
    iters = map(AbstractPPL.varname_and_value_leaves, keys(vals), values(vals))
    vns_vals_iter = mapreduce(collect, vcat, iters)
    syms = map(Symbol ∘ first, vns_vals_iter)
    split_vals = map(last, vns_vals_iter)
    return ModeResult(
        NamedArrays.NamedArray(split_vals, syms),
        solution,
        -solution.objective,
        log_density,
        vals,
        linked,
        estimator,
    )
end

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
function has_generic_constraints(c::ModeEstimationConstraints)
    return (c.cons !== nothing || c.lcons !== nothing || c.ucons !== nothing)
end
has_constraints(c) = has_box_constraints(c) || has_generic_constraints(c)

"""
    generate_initial_params(model::DynamicPPL.Model, initial_params, constraints)

Generate an initial value for the optimization problem.

If `initial_params` is not `nothing`, a copy of it is returned. Otherwise initial parameter
values are generated either by sampling from the prior (if no constraints are present) or
uniformly from the box constraints. If generic constraints are set, an error is thrown.
"""
function generate_initial_params(model::DynamicPPL.Model, initial_params, constraints)
    if initial_params === nothing && has_generic_constraints(constraints)
        throw(
            ArgumentError(
                "You must provide an initial value when using generic constraints."
            ),
        )
    end

    return if initial_params !== nothing
        copy(initial_params)
    elseif has_box_constraints(constraints)
        [
            rand(Distributions.Uniform(lower, upper)) for
            (lower, upper) in zip(constraints.lb, constraints.ub)
        ]
    else
        rand(Vector, model)
    end
end

function default_solver(constraints::ModeEstimationConstraints)
    return if has_generic_constraints(constraints)
        OptimizationOptimJL.IPNewton()
    else
        OptimizationOptimJL.LBFGS()
    end
end

"""
    OptimizationProblem(log_density::OptimLogDensity, initial_params::AbstractVector, adtype, constraints)

Create an `OptimizationProblem` for the objective function defined by `log_density`.

Note that the adtype parameter here overrides any adtype parameter the
OptimLogDensity was constructed with.
"""
function Optimization.OptimizationProblem(
    log_density::OptimLogDensity, initial_params::AbstractVector, adtype, constraints
)
    # Note that OptimLogDensity is a callable that evaluates the model with given
    # parameters. Hence we can use it in the objective function as below.
    f = Optimization.OptimizationFunction(log_density, adtype; cons=constraints.cons)
    prob = if !has_constraints(constraints)
        Optimization.OptimizationProblem(f, initial_params)
    else
        Optimization.OptimizationProblem(
            f,
            initial_params;
            lcons=constraints.lcons,
            ucons=constraints.ucons,
            lb=constraints.lb,
            ub=constraints.ub,
        )
    end
    return prob
end

# Note that we use `getlogjoint` rather than `getlogjoint_internal`: this is intentional,
# because even though the VarInfo may be linked, the optimisation target should not take the
# Jacobian term into account.
_choose_getlogdensity(::MAP) = DynamicPPL.getlogjoint
_choose_getlogdensity(::MLE) = DynamicPPL.getloglikelihood

"""
    estimate_mode(
        model::DynamicPPL.Model,
        estimator::ModeEstimator,
        [solver];
        kwargs...
    )

Find the mode of the probability distribution of a model.

Under the hood this function calls `Optimization.solve`.

# Arguments
- `model::DynamicPPL.Model`: The model for which to estimate the mode.
- `estimator::ModeEstimator`: Can be either `MLE()` for maximum likelihood estimation or
    `MAP()` for maximum a posteriori estimation.
- `solver=nothing`. The optimization algorithm to use. Optional. Can be any solver
    recognised by Optimization.jl. If omitted a default solver is used: LBFGS, or IPNewton
    if non-box constraints are present.

# Keyword arguments
- `check_model::Bool=true`: If true, the model is checked for errors before
    optimisation begins.
- `initial_params::Union{AbstractVector,Nothing}=nothing`: Initial value for the
    optimization. Optional, unless non-box constraints are specified. If omitted it is
    generated by either sampling from the prior distribution or uniformly from the box
    constraints, if any.
- `adtype::AbstractADType=AutoForwardDiff()`: The automatic differentiation type to use.
- Keyword arguments `lb`, `ub`, `cons`, `lcons`, and `ucons` define constraints for the
    optimization problem. Please see [`ModeEstimationConstraints`](@ref) for more details.
- Any extra keyword arguments are passed to `Optimization.solve`.
"""
function estimate_mode(
    model::DynamicPPL.Model,
    estimator::ModeEstimator,
    solver=nothing;
    check_model::Bool=true,
    initial_params=nothing,
    adtype=ADTypes.AutoForwardDiff(),
    cons=nothing,
    lcons=nothing,
    ucons=nothing,
    lb=nothing,
    ub=nothing,
    kwargs...,
)
    if check_model
        new_model = DynamicPPL.setleafcontext(model, DynamicPPL.InitContext())
        DynamicPPL.check_model(new_model, DynamicPPL.VarInfo(); error_on_failure=true)
    end

    constraints = ModeEstimationConstraints(lb, ub, cons, lcons, ucons)
    initial_params = generate_initial_params(model, initial_params, constraints)
    if solver === nothing
        solver = default_solver(constraints)
    end

    # Set its VarInfo to the initial parameters.
    # TODO(penelopeysm): Unclear if this is really needed? Any time that logp is calculated
    # (using `LogDensityProblems.logdensity(ldf, x)`) the parameters in the
    # varinfo are completely ignored. The parameters only matter if you are calling evaluate!!
    # directly on the fields of the LogDensityFunction
    vi = DynamicPPL.VarInfo(model)
    vi = DynamicPPL.unflatten(vi, initial_params)

    # Link the varinfo if needed.
    # TODO(mhauru) We currently couple together the questions of whether the user specified
    # bounds/constraints and whether we transform the objective function to an
    # unconstrained space. These should be separate concerns, but for that we need to
    # implement getting the bounds of the prior distributions.
    optimise_in_unconstrained_space = !has_constraints(constraints)
    if optimise_in_unconstrained_space
        vi = DynamicPPL.link(vi, model)
    end
    # Re-extract initial parameters (which may now be linked).
    initial_params = vi[:]

    # Note that we don't need adtype here, because it's specified inside the
    # OptimizationProblem
    getlogdensity = _choose_getlogdensity(estimator)
    ldf = DynamicPPL.LogDensityFunction(model, getlogdensity, vi)
    # Create an OptimLogDensity object that can be used to evaluate the objective function,
    # i.e. the negative log density.
    log_density = OptimLogDensity(ldf)

    prob = Optimization.OptimizationProblem(
        log_density, initial_params, adtype, constraints
    )
    solution = Optimization.solve(prob, solver; kwargs...)
    # TODO(mhauru) We return a ModeResult for compatibility with the older Optim.jl
    # interface. Might we want to break that and develop a better return type?
    return ModeResult(log_density, solution, optimise_in_unconstrained_space, estimator)
end

"""
    maximum_a_posteriori(
        model::DynamicPPL.Model,
        [solver];
        kwargs...
    )

Find the maximum a posteriori estimate of a model.

This is a convenience function that calls `estimate_mode` with `MAP()` as the estimator.
Please see the documentation of [`Turing.Optimisation.estimate_mode`](@ref) for more
details.
"""
function maximum_a_posteriori(model::DynamicPPL.Model, args...; kwargs...)
    return estimate_mode(model, MAP(), args...; kwargs...)
end

"""
    maximum_likelihood(
        model::DynamicPPL.Model,
        [solver];
        kwargs...
    )

Find the maximum likelihood estimate of a model.

This is a convenience function that calls `estimate_mode` with `MLE()` as the estimator.
Please see the documentation of [`Turing.Optimisation.estimate_mode`](@ref) for more
details.
"""
function maximum_likelihood(model::DynamicPPL.Model, args...; kwargs...)
    return estimate_mode(model, MLE(), args...; kwargs...)
end

include("stats.jl")

end
