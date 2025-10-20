module TuringAbstractMCMCTests

using AbstractMCMC: AbstractMCMC
using DynamicPPL: DynamicPPL
using Random: AbstractRNG
using Test: @test, @testset, @test_throws
using Turing

@testset "Initial parameters" begin
    # Dummy algorithm that just returns initial value and does not perform any sampling
    abstract type OnlyInit <: AbstractMCMC.AbstractSampler end
    struct OnlyInitDefault <: OnlyInit end
    struct OnlyInitUniform <: OnlyInit end
    Turing.Inference.init_strategy(::OnlyInitUniform) = InitFromUniform()
    function Turing.Inference.initialstep(
        rng::AbstractRNG,
        model::DynamicPPL.Model,
        ::OnlyInit,
        vi::DynamicPPL.VarInfo=DynamicPPL.VarInfo(rng, model);
        kwargs...,
    )
        return vi, nothing
    end

    @testset "init_strategy" begin
        # check that the default init strategy is prior
        @test Turing.Inference.init_strategy(OnlyInitDefault()) == InitFromPrior()
        @test Turing.Inference.init_strategy(OnlyInitUniform()) == InitFromUniform()
    end

    for spl in (OnlyInitDefault(), OnlyInitUniform())
        # model with one variable: initialization p = 0.2
        @model function coinflip()
            p ~ Beta(1, 1)
            return 10 ~ Binomial(25, p)
        end
        model = coinflip()
        lptrue = logpdf(Binomial(25, 0.2), 10)
        let inits = InitFromParams((; p=0.2))
            chain = sample(model, spl, 1; initial_params=inits, progress=false)
            @test chain[1].metadata.p.vals == [0.2]
            @test DynamicPPL.getlogjoint(chain[1]) == lptrue

            # parallel sampling
            chains = sample(
                model,
                spl,
                MCMCThreads(),
                1,
                10;
                initial_params=fill(inits, 10),
                progress=false,
            )
            for c in chains
                @test c[1].metadata.p.vals == [0.2]
                @test DynamicPPL.getlogjoint(c[1]) == lptrue
            end
        end

        # check that Vector no longer works
        @test_throws ArgumentError sample(
            model, spl, 1; initial_params=[4, -1], progress=false
        )
        @test_throws ArgumentError sample(
            model, spl, 1; initial_params=[missing, -1], progress=false
        )

        # model with two variables: initialization s = 4, m = -1
        @model function twovars()
            s ~ InverseGamma(2, 3)
            return m ~ Normal(0, sqrt(s))
        end
        model = twovars()
        lptrue = logpdf(InverseGamma(2, 3), 4) + logpdf(Normal(0, 2), -1)
        for inits in (
            InitFromParams((s=4, m=-1)),
            (s=4, m=-1),
            InitFromParams(Dict(@varname(s) => 4, @varname(m) => -1)),
            Dict(@varname(s) => 4, @varname(m) => -1),
        )
            chain = sample(model, spl, 1; initial_params=inits, progress=false)
            @test chain[1].metadata.s.vals == [4]
            @test chain[1].metadata.m.vals == [-1]
            @test DynamicPPL.getlogjoint(chain[1]) == lptrue

            # parallel sampling
            chains = sample(
                model,
                spl,
                MCMCThreads(),
                1,
                10;
                initial_params=fill(inits, 10),
                progress=false,
            )
            for c in chains
                @test c[1].metadata.s.vals == [4]
                @test c[1].metadata.m.vals == [-1]
                @test DynamicPPL.getlogjoint(c[1]) == lptrue
            end
        end

        # set only m = -1
        for inits in (
            InitFromParams((; s=missing, m=-1)),
            InitFromParams(Dict(@varname(s) => missing, @varname(m) => -1)),
            (; s=missing, m=-1),
            Dict(@varname(s) => missing, @varname(m) => -1),
            InitFromParams((; m=-1)),
            InitFromParams(Dict(@varname(m) => -1)),
            (; m=-1),
            Dict(@varname(m) => -1),
        )
            chain = sample(model, spl, 1; initial_params=inits, progress=false)
            @test !ismissing(chain[1].metadata.s.vals[1])
            @test chain[1].metadata.m.vals == [-1]

            # parallel sampling
            chains = sample(
                model,
                spl,
                MCMCThreads(),
                1,
                10;
                initial_params=fill(inits, 10),
                progress=false,
            )
            for c in chains
                @test !ismissing(c[1].metadata.s.vals[1])
                @test c[1].metadata.m.vals == [-1]
            end
        end
    end
end

end # module
