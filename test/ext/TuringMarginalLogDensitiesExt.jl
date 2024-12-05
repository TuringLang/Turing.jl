module TuringMarginalLogDensitiesExt

using Turing, MarginalLogDensities, Test

@testset "MarginalLogDensities" begin
    # Simple test case.
    @model function demo()
        x ~ Normal(0, 1)
        y ~ Normal(x, 1)
    end
    model = demo();
    # Marginalize out `x`.
    marginalized = marginalize(model, [@varname(x)]);
    # Compute the marginal log-density of `y = 0.0`.
    @test marginalized([0.0]) ≈ logpdf(Normal(0, √2), 0.0) atol=2e-1
end
