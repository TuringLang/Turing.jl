module TuringMarginalLogDensitiesTest

using Turing, MarginalLogDensities, Test

@testset "MarginalLogDensities" begin
    # Simple test case.
    @model function demo()
        x ~ MvNormal(zeros(2), [1, 1])
        y ~ Normal(0, 1)
    end
    model = demo();
    # Marginalize out `x`.
    marginalized = marginalize(model, [@varname(x)]);
    marginalized = marginalize(model, [:x]);
    # Compute the marginal log-density of `y = 0.0`.
    @test marginalized([0.0]) ≈ logpdf(Normal(0, 1), 0.0) atol=2e-1
end

end