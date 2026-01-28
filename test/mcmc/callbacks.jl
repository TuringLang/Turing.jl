module CallbacksTests

using Test, Turing, AbstractMCMC, Random, Distributions, LinearAlgebra

@model function test_normals()
    x ~ Normal()
    return y ~ MvNormal(zeros(3), I)
end

@testset "AbstractMCMC Callbacks Interface" begin
    rng = Random.default_rng()
    model = test_normals()

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
        @testset "$name" begin
            t1, s1 = AbstractMCMC.step(
                rng, model, sampler; initial_params=Turing.Inference.init_strategy(sampler)
            )

            # ParamsWithStats returns named params (not θ[i])
            pws = AbstractMCMC.ParamsWithStats(
                model, sampler, t1, s1; params=true, stats=true
            )
            pairs_dict = Dict(k => v for (k, v) in Base.pairs(pws))
            # Keys are Symbols since ParamsWithStats stores NamedTuple internally
            @test haskey(pairs_dict, Symbol("x"))
            @test haskey(pairs_dict, Symbol("y"))
            @test pairs_dict[Symbol("y")] isa AbstractVector
            @test length(pairs_dict[Symbol("y")]) == 3

            # Check stats contain lp
            @test haskey(pairs_dict, :lp) || haskey(pairs_dict, :logjoint)
        end
    end

    # NUTS second step has full AHMC transition metrics
    @testset "NUTS Transition Metrics" begin
        sampler = NUTS(10, 0.65)
        t1, s1 = AbstractMCMC.step(
            rng, model, sampler; initial_params=Turing.Inference.init_strategy(sampler)
        )
        t2, s2 = AbstractMCMC.step(rng, model, sampler, s1)

        pws = AbstractMCMC.ParamsWithStats(model, sampler, t2, s2; params=true, stats=true)
        pairs_dict = Dict(k => v for (k, v) in Base.pairs(pws))

        # Keys are Symbols from NamedTuple
        @test haskey(pairs_dict, :tree_depth)
        @test haskey(pairs_dict, :n_steps)
        @test haskey(pairs_dict, :acceptance_rate)
        @test haskey(pairs_dict, :hamiltonian_energy)
    end
end

end
