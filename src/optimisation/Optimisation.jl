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

export maximum_a_posteriori, maximum_likelihood
# The MAP and MLE exports are only needed for the Optim.jl interface.
export MAP, MLE

"""
    ModeEstimator

An abstract type to mark whether mode estimation is to be done with maximum a posteriori
(MAP) or maximum likelihood estimation (MLE). This is only needed for the Optim.jl interface.
"""
abstract type ModeEstimator end

"""
    MLE <: ModeEstimator

Concrete type for maximum likelihood estimation. Only used for the Optim.jl interface.
"""
struct MLE <: ModeEstimator end

"""
    MAP <: ModeEstimator

Concrete type for maximum a posteriori estimation. Only used for the Optim.jl interface.
"""
struct MAP <: ModeEstimator end

"""
    OptimLogDensity{
        M<:DynamicPPL.Model,
        F<:Function,
        V<:DynamicPPL.AbstractVarInfo,
        AD<:ADTypes.AbstractADType,
    }

A struct that wraps a single LogDensityFunction. Can be invoked either using

```julia
OptimLogDensity(model, varinfo; adtype=adtype)
```

or

```julia
OptimLogDensity(model; adtype=adtype)
```

If not specified, `adtype` defaults to `AutoForwardDiff()`.

An OptimLogDensity does not, in itself, obey the LogDensityProblems interface.
Thus, if you want to calculate the log density of its contents at the point
`z`, you should manually call

```julia
LogDensityProblems.logdensity(f.ldf, z)
```

However, it is a callable object which returns the *negative* log density of
the underlying LogDensityFunction at the point `z`. This is done to satisfy
the Optim.jl interface.

```julia
optim_ld = OptimLogDensity(model, varinfo)
optim_ld(z)  # returns -logp
```
"""
struct OptimLogDensity{L<:DynamicPPL.LogDensityFunction}
    ldf::L

    function OptimLogDensity(
        model::DynamicPPL.Model,
        getlogdensity::Function,
        vi::DynamicPPL.AbstractVarInfo;
        adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE,
    )
        ldf = DynamicPPL.LogDensityFunction(model, getlogdensity, vi; adtype=adtype)
        return new{typeof(ldf)}(ldf)
    end
    function OptimLogDensity(
        model::DynamicPPL.Model,
        getlogdensity::Function;
        adtype::ADTypes.AbstractADType=Turing.DEFAULT_ADTYPE,
    )
        # No varinfo
        return OptimLogDensity(
            model,
            getlogdensity,
            DynamicPPL.ldf_default_varinfo(model, getlogdensity);
            adtype=adtype,
        )
    end
end

"""
    (f::OptimLogDensity)(z)
    (f::OptimLogDensity)(z, _)

Evaluate the negative log joint or log likelihood at the array `z`. Which one is evaluated
depends on the context of `f`.

Any second argument is ignored. The two-argument method only exists to match interface the
required by Optimization.jl.
"""
(f::OptimLogDensity)(z::AbstractVector) = -LogDensityProblems.logdensity(f.ldf, z)
(f::OptimLogDensity)(z, _) = f(z)

# NOTE: The format of this function is dictated by Optim. The first argument sets whether to
# compute the function value, the second whether to compute the gradient (and stores the
# gradient). The last one is the actual argument of the objective function.
function (f::OptimLogDensity)(F, G, z)
    if G !== nothing
        # Calculate log joint and its gradient.
        logp, ∇logp = LogDensityProblems.logdensity_and_gradient(f.ldf, z)

        # Save the negative gradient to the pre-allocated array.
        copyto!(G, -∇logp)

        # If F is something, the negative log joint is requested as well.
        # We have already computed it as a by-product above and hence return it directly.
        if F !== nothing
            return -logp
        end
    end

    # Only negative log joint requested but no gradient.
    if F !== nothing
        return -LogDensityProblems.logdensity(f.ldf, z)
    end

    return nothing
end

"""
    ModeResult{
        V<:NamedArrays.NamedArray,
        M<:NamedArrays.NamedArray,
        O<:Optim.MultivariateOptimizationResults,
        S<:NamedArrays.NamedArray,
        P<:AbstractDict{VarName,Any}
    }

A wrapper struct to store various results from a MAP or MLE estimation.

## Fields

$(TYPEDFIELDS)
"""
struct ModeResult{
    V<:NamedArrays.NamedArray,
    O<:Any,
    M<:OptimLogDensity,
    P<:AbstractDict{AbstractPPL.VarName,Any},
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

# Various StatsBase methods for ModeResult

"""
    StatsBase.coeftable(m::ModeResult; level::Real=0.95, numerrors_warnonly::Bool=true)


Return a table with coefficients and related statistics of the model. level determines the
level for confidence intervals (by default, 95%).

In case the `numerrors_warnonly` argument is true (the default) numerical errors encountered
during the computation of the standard errors will be caught and reported in an extra
"Error notes" column.
"""
function StatsBase.coeftable(m::ModeResult; level::Real=0.95, numerrors_warnonly::Bool=true)
    # Get columns for coeftable.
    terms = string.(StatsBase.coefnames(m))
    estimates = m.values.array[:, 1]
    # If numerrors_warnonly is true, and if either the information matrix is singular or has
    # negative entries on its diagonal, then `notes` will be a list of strings for each
    # value in `m.values`, explaining why the standard error is NaN.
    notes = nothing
    local stderrors
    if numerrors_warnonly
        infmat = StatsBase.informationmatrix(m)
        local vcov
        try
            vcov = inv(infmat)
        catch e
            if isa(e, LinearAlgebra.SingularException)
                stderrors = fill(NaN, length(m.values))
                notes = fill("Information matrix is singular", length(m.values))
            else
                rethrow(e)
            end
        else
            vars = LinearAlgebra.diag(vcov)
            stderrors = eltype(vars)[]
            if any(x -> x < 0, vars)
                notes = []
            end
            for var in vars
                if var >= 0
                    push!(stderrors, sqrt(var))
                    if notes !== nothing
                        push!(notes, "")
                    end
                else
                    push!(stderrors, NaN)
                    if notes !== nothing
                        push!(notes, "Negative variance")
                    end
                end
            end
        end
    else
        stderrors = StatsBase.stderror(m)
    end
    zscore = estimates ./ stderrors
    p = map(z -> StatsAPI.pvalue(Distributions.Normal(), z; tail=:both), zscore)

    # Confidence interval (CI)
    q = Statistics.quantile(Distributions.Normal(), (1 + level) / 2)
    ci_low = estimates .- q .* stderrors
    ci_high = estimates .+ q .* stderrors

    level_ = 100 * level
    level_percentage = isinteger(level_) ? Int(level_) : level_

    cols = Vector[estimates, stderrors, zscore, p, ci_low, ci_high]
    colnms = [
        "Coef.",
        "Std. Error",
        "z",
        "Pr(>|z|)",
        "Lower $(level_percentage)%",
        "Upper $(level_percentage)%",
    ]
    if notes !== nothing
        push!(cols, notes)
        push!(colnms, "Error notes")
    end
    return StatsBase.CoefTable(cols, colnms, terms)
end

function StatsBase.informationmatrix(
    m::ModeResult; hessian_function=ForwardDiff.hessian, kwargs...
)
    # Calculate Hessian and information matrix.

    # Convert the values to their unconstrained states to make sure the
    # Hessian is computed with respect to the untransformed parameters.
    old_ldf = m.f.ldf
    linked = DynamicPPL.is_transformed(old_ldf.varinfo)
    if linked
        new_vi = DynamicPPL.invlink!!(old_ldf.varinfo, old_ldf.model)
        new_f = OptimLogDensity(
            old_ldf.model, old_ldf.getlogdensity, new_vi; adtype=old_ldf.adtype
        )
        m = Accessors.@set m.f = new_f
    end

    # Calculate the Hessian, which is the information matrix because the negative of the log
    # likelihood was optimized
    varnames = StatsBase.coefnames(m)
    info = hessian_function(m.f, m.values.array[:, 1])

    # Link it back if we invlinked it.
    if linked
        invlinked_ldf = m.f.ldf
        new_vi = DynamicPPL.link!!(invlinked_ldf.varinfo, invlinked_ldf.model)
        new_f = OptimLogDensity(
            invlinked_ldf.model, old_ldf.getlogdensity, new_vi; adtype=invlinked_ldf.adtype
        )
        m = Accessors.@set m.f = new_f
    end

    return NamedArrays.NamedArray(info, (varnames, varnames))
end

StatsBase.coef(m::ModeResult) = m.values
StatsBase.coefnames(m::ModeResult) = names(m.values)[1]
StatsBase.params(m::ModeResult) = StatsBase.coefnames(m)
StatsBase.vcov(m::ModeResult) = inv(StatsBase.informationmatrix(m))
StatsBase.loglikelihood(m::ModeResult) = m.lp

"""
    Base.get(m::ModeResult, var_symbol::Symbol)
    Base.get(m::ModeResult, var_symbols::AbstractVector{Symbol})

Return the values of all the variables with the symbol(s) `var_symbol` in the mode result
`m`. The return value is a `NamedTuple` with `var_symbols` as the key(s). The second
argument should be either a `Symbol` or a vector of `Symbol`s.
"""
function Base.get(m::ModeResult, var_symbols::AbstractVector{Symbol})
    log_density = m.f.ldf
    # Get all the variable names in the model. This is the same as the list of keys in
    # m.values, but they are more convenient to filter when they are VarNames rather than
    # Symbols.
    vals_dict = Turing.Inference.getparams(log_density.model, log_density.varinfo)
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
    ModeResult(log_density::OptimLogDensity, solution::SciMLBase.OptimizationSolution)

Create a `ModeResult` for a given `log_density` objective and a `solution` given by `solve`.

`Optimization.solve` returns its own result type. This function converts that into the
richer format of `ModeResult`. It also takes care of transforming them back to the original
parameter space in case the optimization was done in a transformed space.
"""
function ModeResult(log_density::OptimLogDensity, solution::SciMLBase.OptimizationSolution)
    varinfo_new = DynamicPPL.unflatten(log_density.ldf.varinfo, solution.u)
    # `getparams` performs invlinking if needed
    vals = Turing.Inference.getparams(log_density.ldf.model, varinfo_new)
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
    OptimizationProblem(log_density::OptimLogDensity, adtype, constraints)

Create an `OptimizationProblem` for the objective function defined by `log_density`.

Note that the adtype parameter here overrides any adtype parameter the
OptimLogDensity was constructed with.
"""
function Optimization.OptimizationProblem(log_density::OptimLogDensity, adtype, constraints)
    # Note that OptimLogDensity is a callable that evaluates the model with given
    # parameters. Hence we can use it in the objective function as below.
    f = Optimization.OptimizationFunction(log_density, adtype; cons=constraints.cons)
    initial_params = log_density.ldf.varinfo[:]
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

    # Create an OptimLogDensity object that can be used to evaluate the objective function,
    # i.e. the negative log density.
    # Note that we use `getlogjoint` rather than `getlogjoint_internal`: this
    # is intentional, because even though the VarInfo may be linked, the
    # optimisation target should not take the Jacobian term into account.
    getlogdensity = estimator isa MAP ? DynamicPPL.getlogjoint : DynamicPPL.getloglikelihood

    # Set its VarInfo to the initial parameters.
    # TODO(penelopeysm): Unclear if this is really needed? Any time that logp is calculated
    # (using `LogDensityProblems.logdensity(ldf, x)`) the parameters in the
    # varinfo are completely ignored. The parameters only matter if you are calling evaluate!!
    # directly on the fields of the LogDensityFunction
    vi = DynamicPPL.ldf_default_varinfo(model, getlogdensity)
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

    log_density = OptimLogDensity(model, getlogdensity, vi)

    prob = Optimization.OptimizationProblem(log_density, adtype, constraints)
    solution = Optimization.solve(prob, solver; kwargs...)
    # TODO(mhauru) We return a ModeResult for compatibility with the older Optim.jl
    # interface. Might we want to break that and develop a better return type?
    return ModeResult(log_density, solution)
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

end
