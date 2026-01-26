import DifferentiationInterface as DI

"""
    get_vector_params(m::ModeResult)

Generates a vectorised form of the optimised parameters stored in the `ModeResult`, along
with the corresponding variable names. These parameters correspond to unlinked space.

This function returns two vectors: the first contains the variable names, and the second
contains the corresponding values.
"""
function get_vector_params(m::ModeResult)
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
    #
    # TODO(penelopeysm): This needs to be done in a "better" way using DynamicPPL
    # functionality. Note that, if https://github.com/TuringLang/DynamicPPL.jl/pull/1178
    # goes through, the value vector can just be generated using
    # `first(DynamicPPL.rand_with_logdensity(m.ldf, InitFromParams(m.params)))`. The
    # varnames, though, are a hot mess. See the TODO below.
    ldf = LogDensityFunction(m.ldf.model)
    vns = Vector{VarName}(undef, LogDensityProblems.dimension(ldf))
    vals = Vector{Any}(undef, LogDensityProblems.dimension(ldf))
    for vn in keys(m.params)
        range = if AbstractPPL.getoptic(vn) === identity
            ldf._iden_varname_ranges[AbstractPPL.getsym(vn)].range
        else
            ldf._varname_ranges[vn].range
        end
        # TODO(penelopeysm): This assumes that tovec and varname_leaves return the same
        # number of sub-elements in the same order --- which is NOT true for things like
        # Cholesky factors --- so this is a bug! We COULD use `varname_and_value_leaves` but
        # that would just be a different kind of bug because LDF uses tovec for
        # vectorisation!! https://github.com/TuringLang/Turing.jl/issues/2734
        val = m.params[vn]
        vns[range] = collect(AbstractPPL.varname_leaves(vn, val))
        vals[range] = DynamicPPL.tovec(val)
    end
    # Concretise
    return [x for x in vns], [x for x in vals]
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
    vns, estimates = get_vector_params(m)
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
    _, x = get_vector_params(m)

    # We can include a check here to make sure that f(x) is in fact the log density at x.
    # This helps guard against potential bugs where `get_vector_params` returns a
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

StatsBase.coef(m::ModeResult) = last(get_vector_params(m))
StatsBase.coefnames(m::ModeResult) = first(get_vector_params(m))
StatsBase.params(m::ModeResult) = StatsBase.coefnames(m)
StatsBase.vcov(m::ModeResult) = inv(StatsBase.informationmatrix(m))
StatsBase.loglikelihood(m::ModeResult) = m.lp
