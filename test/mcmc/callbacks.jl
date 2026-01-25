module CallbacksTests

using Test, Turing, AbstractMCMC, Random, Distributions, LinearAlgebra

# Simple model that works for all samplers (ESS requires Normal distributions)
@model function test_normals()
    x ~ Normal()
    y ~ MvNormal(zeros(3), I)
end

@testset "AbstractMCMC Callbacks Interface" begin
    rng = Random.default_rng()
    model = test_normals()

    # All samplers use the same model (4 params: x + y[1:3])
    samplers = [
        ("NUTS", NUTS(10, 0.65)),
        ("HMC", HMC(0.1, 5)),
        ("MH", MH()),
        ("ESS", ESS()),
        ("Gibbs", Gibbs(:x => HMC(0.1, 5), :y => MH())),
        ("SGHMC", SGHMC(; learning_rate=0.01, momentum_decay=1e-2)),
        ("PG", PG(10)),
    ]

    for (name, sampler) in samplers
        @testset "$name Interface" begin
            transition, state = AbstractMCMC.step(
                rng, model, sampler; initial_params=Turing.Inference.init_strategy(sampler)
            )

            # Should return a flat vector of Reals (unconstrained)
            params = AbstractMCMC.getparams(state)
            @test params isa AbstractVector{<:Real}
            @test length(params) == 4  # x (1) + y (3)

            # Should return a NamedTuple with at least log probability (:lp)
            stats = AbstractMCMC.getstats(state)
            @test stats isa NamedTuple
            @test haskey(stats, :lp)
            @test stats.lp isa Real
            @test isfinite(stats.lp)  # Should be a valid log probability
        end
    end
end

end
