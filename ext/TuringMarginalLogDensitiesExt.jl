module TuringMarginalLogDensitiesExt

using Turing: Turing, DynamicPPL
using Turing.Inference: LogDensityProblems
using MarginalLogDensities: MarginalLogDensities

# Use a struct for this to avoid closure overhead.
struct Drop2ndArgAndFlipSign{F}
    f::F
end

(f::Drop2ndArgAndFlipSign)(x, _) = -f.f(x)

function Turing.marginalize(
    model::DynamicPPL.Model,
    varnames::Vector,
    method::MarginalLogDensities.AbstractMarginalizer=MarginalLogDensities.LaplaceApprox(),
)
    # Determine the indices for the variables to marginalise out.
    varinfo = DynamicPPL.typed_varinfo(model)
    varindices = DynamicPPL.vector_getranges(varinfo, varnames)
    # Construct the marginal log-density model.
    # Use linked `varinfo` to that we're working in unconstrained space and `OptimizationContext` to ensure
    # that the log-abs-det jacobian terms are not included.
    context = Turing.Optimisation.OptimizationContext(DynamicPPL.leafcontext(model.context))
    varinfo_linked = DynamicPPL.link(varinfo, model)
    f = Base.Fix1(
        LogDensityProblems.logdensity,
        DynamicPPL.LogDensityFunction(varinfo_linked, model, context),
    )
    # HACK: need the sign-flip here because `OptimizationContext` is a hacky impl which
    # represent the _negative_ log-density.
    mdl = MarginalLogDensities.MarginalLogDensity(
        Drop2ndArgAndFlipSign(f), varinfo_linked[:], varindices, (), method
    )
    return mdl
end

end
