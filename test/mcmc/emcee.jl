module EmceeTests

using ..Models: gdemo_default
using ..NumericalTests: check_gdemo
using Distributions: sample
using DynamicPPL: DynamicPPL
using Random: Random, Xoshiro
using StableRNGs: StableRNG
using FlexiChains: FlexiChains
using LinearAlgebra: I
using MCMCChains: MCMCChains
using Test: @test, @test_throws, @testset
using Turing

@testset "emcee.jl" begin
    @testset "gdemo" begin
        n_samples = 1000
        n_walkers = 250
        spl = Emcee(n_walkers, 2.0)
        chain = sample(StableRNG(9876), gdemo_default, spl, n_samples)
        check_gdemo(chain)
    end

    @testset "memory usage with large number of iterations" begin
        # https://github.com/TuringLang/Turing.jl/pull/1976
        @info "Testing emcee with large number of iterations"
        spl = Emcee(10, 2.0)
        n_samples = 10_000
        chain = sample(StableRNG(5), gdemo_default, spl, n_samples)
        check_gdemo(chain)
    end

    @testset "initial parameters" begin
        nwalkers = 250
        spl = Emcee(nwalkers, 2.0)

        rng1 = Xoshiro(1234)
        chain1 = sample(rng1, gdemo_default, spl, 1)
        rng2 = Xoshiro(1234)
        chain2 = sample(rng2, gdemo_default, spl, 1)
        @test FlexiChains.has_same_data(chain1, chain2)

        initial_nt = DynamicPPL.InitFromParams((s=2.0, m=1.0))
        # Initial parameters have to be specified for every walker
        @test_throws ArgumentError sample(gdemo_default, spl, 1; initial_params=initial_nt)
        @test_throws r"must be a vector of" sample(
            gdemo_default, spl, 1; initial_params=initial_nt
        )

        # Initial parameters
        chain = sample(gdemo_default, spl, 1; initial_params=fill(initial_nt, nwalkers))
        @test chain[:s] == fill(2.0, 1, nwalkers)
        @test chain[:m] == fill(1.0, 1, nwalkers)
    end

    @testset "chain_type" begin
        # `bundle_samples` declared `::Type{MCMCChains.Chains}` positionally where the caller
        # passes it as a keyword, so this silently fell through to the generic method and
        # returned the raw transitions.
        chain = sample(
            StableRNG(7), gdemo_default, Emcee(4), 5; chain_type=MCMCChains.Chains
        )
        @test chain isa MCMCChains.Chains
    end

    @testset "model whose dimension depends on its own draws" begin
        # Walkers initialised from the prior land in different branches, so they have
        # different numbers of parameters and the stretch move has no line to move along.
        @model function branchy()
            b ~ Bernoulli(0.5)
            x ~ Normal(0, 1)
            if b
                y ~ Normal(0, 1)
                1.0 ~ Normal(x + y, 1)
            else
                1.0 ~ Normal(x, 1)
            end
        end
        @test_throws(
            "do not have the same parameter layout",
            sample(StableRNG(5), branchy(), Emcee(10), 5),
        )
        # Two branches can hold different variables and the same total dimension, which a
        # comparison of lengths passed; sampling then threw `KeyError: key y not found`.
        @model function samelen()
            b ~ Bernoulli(0.5)
            if b == 1
                x ~ Normal(0, 1)
                1.0 ~ Normal(x, 1)
            else
                y ~ Normal(100, 1)
                1.0 ~ Normal(y, 1)
            end
        end
        @test_throws(
            "do not have the same parameter layout",
            sample(StableRNG(468), samelen(), Emcee(10, 2.0), 5),
        )
        # And the converse: the same variable NAMES with a different width. Comparing names
        # alone let this through to a `DimensionMismatch` from inside the stretch move, so both
        # the names and the total length are compared.
        @model function samenames()
            n ~ Normal()
            x ~ MvNormal(zeros(n > 0 ? 2 : 3), I)
            return 0.5 ~ Normal(sum(x), 1.0)
        end
        @test_throws(
            "do not have the same parameter layout",
            sample(StableRNG(468), samenames(), Emcee(20, 2.0), 5),
        )
    end
end

end
