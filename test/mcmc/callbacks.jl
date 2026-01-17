module TuringCallbacksTests

using Test: @test, @testset
using Turing
using AbstractMCMC: AbstractMCMC
using Random: Random
using DynamicPPL: DynamicPPL

using ..Models: gdemo_default

@testset "AbstractMCMC Callbacks Interface" begin
    @testset "getparams from states" begin
        @testset "HMCState getparams" begin
            Random.seed!(42)
            chain = sample(gdemo_default, NUTS(100, 0.65), 10; progress=false)

            rng = Random.default_rng()
            transition, state = AbstractMCMC.step(
                rng,
                gdemo_default,
                NUTS(100, 0.65);
                initial_params=Turing.Inference.init_strategy(NUTS(100, 0.65)),
            )

            params = AbstractMCMC.getparams(state)
            @test params isa Vector
            @test length(params) >= 2
            @test all(p -> p isa Pair{String,<:Any}, params)
        end

        @testset "MHState getparams" begin
            Random.seed!(42)
            transition, state = AbstractMCMC.step(
                Random.default_rng(),
                gdemo_default,
                MH();
                initial_params=Turing.Inference.init_strategy(MH()),
            )

            params = AbstractMCMC.getparams(state)
            @test params isa Vector
            @test length(params) >= 2
            @test all(p -> p isa Pair{String,<:Any}, params)
        end

        @testset "ESS getparams (VarInfo as state)" begin
            # ESS only works with Gaussian priors - need simple model
            @model function gaussian_model()
                m ~ Normal(0, 1)
                return nothing
            end

            Random.seed!(42)
            transition, state = AbstractMCMC.step(
                Random.default_rng(),
                gaussian_model(),
                ESS();
                initial_params=Turing.Inference.init_strategy(ESS()),
            )

            @test state isa DynamicPPL.AbstractVarInfo
            params = AbstractMCMC.getparams(state)
            @test params isa Vector
            @test length(params) >= 1
        end
    end

    @testset "getstats from states" begin
        @testset "HMCState getstats" begin
            Random.seed!(42)
            transition, state = AbstractMCMC.step(
                Random.default_rng(),
                gdemo_default,
                NUTS(100, 0.65);
                initial_params=Turing.Inference.init_strategy(NUTS(100, 0.65)),
            )

            stats = AbstractMCMC.getstats(state)
            @test stats isa NamedTuple
            @test haskey(stats, :lp)
            @test stats.lp isa Real
        end

        @testset "MHState getstats" begin
            Random.seed!(42)
            transition, state = AbstractMCMC.step(
                Random.default_rng(),
                gdemo_default,
                MH();
                initial_params=Turing.Inference.init_strategy(MH()),
            )

            stats = AbstractMCMC.getstats(state)
            @test stats isa NamedTuple
            @test haskey(stats, :lp)
            @test stats.lp isa Real
        end

        @testset "ESS getstats (VarInfo)" begin
            @model function gaussian_model()
                m ~ Normal(0, 1)
                return nothing
            end

            Random.seed!(42)
            transition, state = AbstractMCMC.step(
                Random.default_rng(),
                gaussian_model(),
                ESS();
                initial_params=Turing.Inference.init_strategy(ESS()),
            )

            stats = AbstractMCMC.getstats(state)
            @test stats isa NamedTuple
            @test haskey(stats, :lp)
            @test stats.lp isa Real
        end
    end

    @testset "ParamsWithStats transitions" begin
        Random.seed!(42)
        transition, _ = AbstractMCMC.step(
            Random.default_rng(),
            gdemo_default,
            NUTS(100, 0.65);
            initial_params=Turing.Inference.init_strategy(NUTS(100, 0.65)),
        )

        @test transition isa DynamicPPL.ParamsWithStats

        stats = AbstractMCMC.getstats(transition)
        @test stats isa NamedTuple

        params = AbstractMCMC.getparams(transition)
        @test params isa Vector
    end

    @testset "hyperparam_metrics" begin
        @testset "NUTS hyperparam_metrics" begin
            sampler = NUTS()
            metrics = AbstractMCMC.hyperparam_metrics(gdemo_default, sampler)
            @test metrics isa Vector{String}
            @test "extras/lp/stat/Mean" in metrics
            @test "extras/acceptance_rate/stat/Mean" in metrics
        end

        @testset "HMC hyperparam_metrics (via Hamiltonian)" begin
            sampler = HMC(0.1, 10)
            metrics = AbstractMCMC.hyperparam_metrics(gdemo_default, sampler)
            @test metrics isa Vector{String}
            @test "extras/lp/stat/Mean" in metrics
        end

        @testset "MH hyperparam_metrics" begin
            sampler = MH()
            metrics = AbstractMCMC.hyperparam_metrics(gdemo_default, sampler)
            @test metrics isa Vector{String}
            @test "extras/lp/stat/Mean" in metrics
        end

        @testset "PG hyperparam_metrics" begin
            sampler = PG(10)
            metrics = AbstractMCMC.hyperparam_metrics(gdemo_default, sampler)
            @test metrics isa Vector{String}
            @test "extras/lp/stat/Mean" in metrics
        end
    end

    @testset "_hyperparams_impl" begin
        @testset "HMC hyperparams" begin
            sampler = HMC(0.1, 10)
            hyperparams = AbstractMCMC._hyperparams_impl(gdemo_default, sampler, nothing)
            @test hyperparams isa Vector

            hp_dict = Dict(hyperparams)
            @test hp_dict["epsilon"] == 0.1
            @test hp_dict["n_leapfrog"] == 10
        end

        @testset "NUTS hyperparams" begin
            sampler = NUTS(200, 0.65)
            hyperparams = AbstractMCMC._hyperparams_impl(gdemo_default, sampler, nothing)
            @test hyperparams isa Vector

            hp_dict = Dict(hyperparams)
            @test hp_dict["n_adapts"] == 200
            @test hp_dict["delta"] == 0.65
            @test haskey(hp_dict, "max_depth")
        end

        @testset "HMCDA hyperparams" begin
            sampler = HMCDA(200, 0.65, 0.3)
            hyperparams = AbstractMCMC._hyperparams_impl(gdemo_default, sampler, nothing)
            @test hyperparams isa Vector

            hp_dict = Dict(hyperparams)
            @test hp_dict["n_adapts"] == 200
            @test hp_dict["delta"] == 0.65
            @test hp_dict["lambda"] == 0.3
        end

        @testset "PG hyperparams" begin
            sampler = PG(10)
            hyperparams = AbstractMCMC._hyperparams_impl(gdemo_default, sampler, nothing)
            @test hyperparams isa Vector

            hp_dict = Dict(hyperparams)
            @test hp_dict["nparticles"] == 10
        end

        @testset "SGHMC hyperparams" begin
            sampler = SGHMC(; learning_rate=0.01, momentum_decay=0.1)
            hyperparams = AbstractMCMC._hyperparams_impl(gdemo_default, sampler, nothing)
            @test hyperparams isa Vector

            hp_dict = Dict(hyperparams)
            @test hp_dict["learning_rate"] == 0.01
            @test hp_dict["momentum_decay"] == 0.1
        end

        @testset "SGLD hyperparams" begin
            sampler = SGLD()
            hyperparams = AbstractMCMC._hyperparams_impl(gdemo_default, sampler, nothing)
            @test hyperparams isa Vector

            hp_dict = Dict(hyperparams)
            @test haskey(hp_dict, "stepsize")
        end
    end
end

end # module
