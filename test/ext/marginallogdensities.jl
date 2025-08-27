module TuringMarginalLogDensitiesTest

using Turing, MarginalLogDensities, Test
using Turing.DynamicPPL: getlogprior, getlogjoint

@testset "MarginalLogDensities" begin
    # Simple test case.
    @model function demo()
        x ~ MvNormal(zeros(2), [1, 1])
        y ~ Normal(0, 1)
    end
    model = demo();
    # Marginalize out `x`.

    for vn in [@varname(x), :x]
        for getlogprob in [getlogprior, getlogjoint]
            marginalized = marginalize(model, [vn], getlogprob, hess_adtype=AutoForwardDiff());
            # Compute the marginal log-density of `y = 0.0`.
            @test marginalized([0.0]) ≈ logpdf(Normal(0, 1), 0.0) atol=1e-5
        end
    end
end

end
