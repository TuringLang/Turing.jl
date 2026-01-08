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
    # This needs to be calculated in unlinked space
    model = m.f.ldf.model
    vi = DynamicPPL.VarInfo(model)
    getlogdensity = _choose_getlogdensity(m.estimator)
    new_optimld = OptimLogDensity(DynamicPPL.LogDensityFunction(model, getlogdensity, vi))

    # Calculate the Hessian, which is the information matrix because the negative of the log
    # likelihood was optimized
    varnames = StatsBase.coefnames(m)
    info = hessian_function(new_optimld, m.values.array[:, 1])
    return NamedArrays.NamedArray(info, (varnames, varnames))
end

StatsBase.coef(m::ModeResult) = m.values
StatsBase.coefnames(m::ModeResult) = names(m.values)[1]
StatsBase.params(m::ModeResult) = StatsBase.coefnames(m)
StatsBase.vcov(m::ModeResult) = inv(StatsBase.informationmatrix(m))
StatsBase.loglikelihood(m::ModeResult) = m.lp
