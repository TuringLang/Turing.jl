module TuringMarginalLogDensitiesExt

using Turing: Turing, DynamicPPL
using Turing.Inference: LogDensityProblems
using MarginalLogDensities: MarginalLogDensities


# Use a struct for this to avoid closure overhead.
struct Drop2ndArgAndFlipSign{F}
    f::F
end

(f::Drop2ndArgAndFlipSign)(x, _) = -f.f(x)

_to_varname(n::Symbol) = DynamicPPL.@varname($n)
_to_varname(n::DynamicPPL.AbstractPPL.VarName) = n

function Turing.marginalize(
    model::DynamicPPL.Model,
    varnames::Vector,
    method::MarginalLogDensities.AbstractMarginalizer = MarginalLogDensities.LaplaceApprox(),
)
    # Determine the indices for the variables to marginalise out.
    varinfo = DynamicPPL.typed_varinfo(model)
    vns = _to_varname.(varnames)
    varindices = reduce(vcat, DynamicPPL.vector_getranges(varinfo, vns))
    # Construct the marginal log-density model.
    # Use linked `varinfo` to that we're working in unconstrained space
    varinfo_linked = DynamicPPL.link(varinfo, model)

    f = Turing.Optimisation.OptimLogDensity(
        model,
        Turing.DynamicPPL.getlogjoint,
        # Turing.DynamicPPL.typed_varinfo(model)
        varinfo_linked
    )

    # HACK: need the sign-flip here because `OptimizationContext` is a hacky impl which
    # represent the _negative_ log-density.
    mdl = MarginalLogDensities.MarginalLogDensity(
        Drop2ndArgAndFlipSign(f), varinfo_linked[:], varindices, (), method
    )
    return mdl
end

end
