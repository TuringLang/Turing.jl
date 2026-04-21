import DifferentiationInterface as DI
import Bijectors.VectorBijectors: optic_vec

"""
    vector_names_and_params(m::ModeResult)

Generates a vectorised form of the optimised parameters stored in the `ModeResult`, along
with the corresponding variable names. These parameters correspond to unlinked space.

This function returns two vectors: the first contains the variable names, and the second
contains the corresponding values.
"""
function vector_names_and_params(m::ModeResult)
    # This function requires some subtlety. We _could_ simply iterate over keys(m.params)
    # and values(m.params), apply AbstractPPL.varname_and_value_leaves to each pair, and
    # then collect them into a vector. *However*, this vector will later have to be used
    # with a LogDensityFunction again! That means that the order of the parameters in the
    # vector must match the order expected by the LogDensityFunction. There's no guarantee
    # that a simple iteration over the Dict will yield the parameters in the correct order.
    #
    # To ensure that this is always the case, we will have to create a LogDensityFunction
    # and then use its stored ranges to extract the parameters in the correct order. This
    # LDF will have to be created in unlinked space.
    ldf = LogDensityFunction(m.ldf.model)
    vns = Vector{VarName}(undef, LogDensityProblems.dimension(ldf))

    # Evaluate the model to get the vectorised parameters in the right order.
    accs = DynamicPPL.OnlyAccsVarInfo(
        DynamicPPL.PriorDistributionAccumulator(), DynamicPPL.VectorParamAccumulator(ldf)
    )
    _, accs = DynamicPPL.init!!(
        ldf.model, accs, InitFromParams(m.params), DynamicPPL.UnlinkAll()
    )
    vector_params = DynamicPPL.get_vector_params(accs)

    # Figure out the VarNames.
    priors = DynamicPPL.get_priors(accs)
    vector_varnames = Vector{VarName}(undef, length(vector_params))
    for (vn, dist) in pairs(priors)
        range = DynamicPPL.get_range_and_transform(ldf, vn).range
        optics = optic_vec(dist)
        # Really shouldn't happen, but catch in case optic_vec isn't properly defined
        if any(isnothing, optics)
            error(
                "The sub-optics for the distribution $dist are not defined. This is a bug in Turing; please file an issue at https://github.com/TuringLang/Turing.jl/issues.",
            )
        end
        vns = map(optic -> AbstractPPL.append_optic(vn, optic), optics)
        vector_varnames[range] = vns
    end

    # Concretise
    return [x for x in vector_varnames], [x for x in vector_params]
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
    vns, estimates = vector_names_and_params(m)
    # Get columns for coeftable.
    terms = string.(vns)
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
                stderrors = fill(NaN, length(estimates))
                notes = fill("Information matrix is singular", length(estimates))
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

"""
    StatsBase.informationmatrix(
        m::ModeResult;
        adtype::ADTypes.AbstractADType=ADTypes.AutoForwardDiff()
    )

Calculate the [Fisher information matrix](https://en.wikipedia.org/wiki/Fisher_information)
for the mode result `m`. This is the negative Hessian of the log-probability at the mode.

The Hessian is calculated using automatic differentiation with the specified `adtype`. By
default this is `ADTypes.AutoForwardDiff()`. In general, however, it may be more efficient
to use forward-over-reverse AD when the model has many parameters. This can be specified
using `DifferentiationInterface.SecondOrder(outer, inner)`; please consult the
[DifferentiationInterface.jl
documentation](https://juliadiff.org/DifferentiationInterface.jl/DifferentiationInterface/stable/explanation/backends/#Second-order)
for more details.
"""
function StatsBase.informationmatrix(
    m::ModeResult; adtype::ADTypes.AbstractADType=ADTypes.AutoForwardDiff()
)
    # This needs to be calculated in unlinked space, regardless of whether the optimization
    # itself was run in linked space.
    model = m.ldf.model
    # We need to get the Hessian for the positive log density.
    ldf = DynamicPPL.LogDensityFunction(model, logprob_func(m.estimator))
    f = Base.Fix1(LogDensityProblems.logdensity, ldf)
    # Then get the vectorised parameters.
    _, x = vector_names_and_params(m)

    # We can include a check here to make sure that f(x) is in fact the log density at x.
    # This helps guard against potential bugs where `vector_names_and_params` returns a
    # wrongly-ordered parameter vector.
    if !isapprox(f(x), m.lp)
        error(
            "The parameter vector extracted from the ModeResult does not match the " *
            "log density stored in the ModeResult. This is a bug in Turing; please " *
            "do file an issue at https://github.com/TuringLang/Turing.jl/issues.",
        )
    end
    return -DI.hessian(f, adtype, x)
end

StatsBase.coef(m::ModeResult) = last(vector_names_and_params(m))
StatsBase.coefnames(m::ModeResult) = first(vector_names_and_params(m))
StatsBase.params(m::ModeResult) = StatsBase.coefnames(m)
StatsBase.vcov(m::ModeResult) = inv(StatsBase.informationmatrix(m))
StatsBase.loglikelihood(m::ModeResult) = m.lp
