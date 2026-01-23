module CallbacksTests

using Test, Turing, AbstractMCMC, Random

if !isdefined(@__MODULE__, :Models)
    include(joinpath(@__DIR__, "..", "test_utils", "models.jl"))
    using .Models: gdemo_default
end

@model function simple_gaussian()
    return x ~ Normal(0, 1)
end

@testset "AbstractMCMC Callbacks Interface" begin
    rng = Random.default_rng()

    samplers = [
        ("NUTS", NUTS(10, 0.65), gdemo_default),
        ("HMC", HMC(0.1, 5), gdemo_default),
        ("MH", MH(), gdemo_default),
        ("ESS", ESS(), simple_gaussian()),
        ("Gibbs", Gibbs(:m => HMC(0.1, 5), :s => MH()), gdemo_default),
        ("SGHMC", SGHMC(; learning_rate=0.01, momentum_decay=1e-2), gdemo_default),
        ("PG", PG(10), gdemo_default),
    ]

    for (name, sampler, model) in samplers
        @testset "$name Interface" begin
            transition, state = AbstractMCMC.step(
                rng, model, sampler; initial_params=Turing.Inference.init_strategy(sampler)
            )

            # Should return a flat vector of Reals (unconstrained)
            params = AbstractMCMC.getparams(state)
            @test params isa Vector{<:Real}
            @test !isempty(params)

            # Should return a NamedTuple with at least log probability (:lp)
            stats = AbstractMCMC.getstats(state)
            @test stats isa NamedTuple
            @test haskey(stats, :lp)
            @test stats.lp isa Real
        end
    end
end

end
