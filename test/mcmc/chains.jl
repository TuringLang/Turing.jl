# Test basic properties of sampling with FlexiChains.jl.
module TuringFlexiChainsTests

using AbstractMCMC: AbstractMCMC
using DynamicPPL: DynamicPPL
using FlexiChains: FlexiChains, FlexiChain, VNChain, Parameter, Extra
using MCMCChains: MCMCChains
using Random: Random, Xoshiro
using Test
using Turing

Turing.setprogress!(false)

# This sampler does nothing (it just stays at the existing state)
struct StaticSampler <: AbstractMCMC.AbstractSampler end
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    ::StaticSampler;
    initial_params::DynamicPPL.AbstractInitStrategy,
    kwargs...,
)
    # generate raw values according to requested initialisation strategy
    vi = DynamicPPL.OnlyAccsVarInfo((DynamicPPL.RawValueAccumulator(false),))
    vi = last(DynamicPPL.init!!(rng, model, vi, initial_params, DynamicPPL.UnlinkAll()))
    vnt = DynamicPPL.get_raw_values(vi)
    return DynamicPPL.ParamsWithStats(vnt, (;)), vnt
end
function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DynamicPPL.Model,
    ::StaticSampler,
    vnt::DynamicPPL.VarNamedTuple;
    kwargs...,
)
    return DynamicPPL.ParamsWithStats(vnt, (;)), vnt
end

@testset verbose = true "chains.jl" begin
    @testset "basic sampling" begin
        @model function demomodel(x)
            m ~ Normal(0, 1.0)
            x ~ Normal(m, 1.0)
            return nothing
        end
        model = demomodel(1.5)

        @testset "single-chain sampling" begin
            @testset "FlexiChains" begin
                chn = sample(model, NUTS(), 100; chain_type=VNChain, verbose=false)
                @test chn isa VNChain
                @test size(chn) == (100, 1)
            end
            @testset "MCMCChains" begin
                chn = sample(
                    model, NUTS(), 100; chain_type=MCMCChains.Chains, verbose=false
                )
                @test chn isa MCMCChains.Chains
                @test size(chn)[1] == 100
                @test size(chn)[3] == 1
            end
        end

        @testset "multi-chain sampling" begin
            @testset "FlexiChains" begin
                chn = sample(
                    model, NUTS(), MCMCSerial(), 100, 3; chain_type=VNChain, verbose=false
                )
                @test chn isa VNChain
                @test size(chn) == (100, 3)
            end
            @testset "MCMCChains" begin
                chn = sample(
                    model,
                    NUTS(),
                    MCMCSerial(),
                    100,
                    3;
                    chain_type=MCMCChains.Chains,
                    verbose=false,
                )
                @test chn isa MCMCChains.Chains
                @test size(chn)[1] == 100
                @test size(chn)[3] == 3
            end
        end

        @testset "rng is respected" begin
            @testset "single-chain" begin
                @testset "FlexiChains" begin
                    chn1 = sample(
                        Xoshiro(468), model, NUTS(), 100; chain_type=VNChain, verbose=false
                    )
                    chn2 = sample(
                        Xoshiro(468), model, NUTS(), 100; chain_type=VNChain, verbose=false
                    )
                    @test FlexiChains.has_same_data(chn1, chn2)
                    chn3 = sample(
                        Xoshiro(469), model, NUTS(), 100; chain_type=VNChain, verbose=false
                    )
                    @test !FlexiChains.has_same_data(chn1, chn3)
                end

                @testset "MCMCChains" begin
                    chn1 = sample(
                        Xoshiro(468),
                        model,
                        NUTS(),
                        100;
                        chain_type=MCMCChains.Chains,
                        verbose=false,
                    )
                    chn2 = sample(
                        Xoshiro(468),
                        model,
                        NUTS(),
                        100;
                        chain_type=MCMCChains.Chains,
                        verbose=false,
                    )
                    @test chn1.value == chn2.value
                    chn3 = sample(
                        Xoshiro(469),
                        model,
                        NUTS(),
                        100;
                        chain_type=MCMCChains.Chains,
                        verbose=false,
                    )
                    @test chn1.value != chn3.value
                end
            end

            @testset "single-chain with seed!" begin
                @testset "FlexiChains" begin
                    Random.seed!(468)
                    chn1 = sample(model, NUTS(), 100; chain_type=VNChain, verbose=false)
                    Random.seed!(468)
                    chn2 = sample(model, NUTS(), 100; chain_type=VNChain, verbose=false)
                    @test FlexiChains.has_same_data(chn1, chn2)
                end

                @testset "MCMCChains" begin
                    Random.seed!(468)
                    chn1 = sample(
                        model, NUTS(), 100; chain_type=MCMCChains.Chains, verbose=false
                    )
                    Random.seed!(468)
                    chn2 = sample(
                        model, NUTS(), 100; chain_type=MCMCChains.Chains, verbose=false
                    )
                    @test chn1.value == chn2.value
                end
            end

            @testset "multi-chain" begin
                @testset "FlexiChains" begin
                    chn1 = sample(
                        Xoshiro(468),
                        model,
                        NUTS(),
                        MCMCSerial(),
                        100,
                        3;
                        chain_type=VNChain,
                        verbose=false,
                    )
                    chn2 = sample(
                        Xoshiro(468),
                        model,
                        NUTS(),
                        MCMCSerial(),
                        100,
                        3;
                        chain_type=VNChain,
                        verbose=false,
                    )
                    @test FlexiChains.has_same_data(chn1, chn2)
                    chn3 = sample(
                        Xoshiro(469),
                        model,
                        NUTS(),
                        MCMCSerial(),
                        100,
                        3;
                        chain_type=VNChain,
                        verbose=false,
                    )
                    @test !FlexiChains.has_same_data(chn1, chn3)
                end

                @testset "MCMCChains" begin
                    chn1 = sample(
                        Xoshiro(468),
                        model,
                        NUTS(),
                        MCMCSerial(),
                        100,
                        3;
                        chain_type=MCMCChains.Chains,
                        verbose=false,
                    )
                    chn2 = sample(
                        Xoshiro(468),
                        model,
                        NUTS(),
                        MCMCSerial(),
                        100,
                        3;
                        chain_type=MCMCChains.Chains,
                        verbose=false,
                    )
                    @test chn1.value == chn2.value
                    chn3 = sample(
                        Xoshiro(469),
                        model,
                        NUTS(),
                        MCMCSerial(),
                        100,
                        3;
                        chain_type=MCMCChains.Chains,
                        verbose=false,
                    )
                    @test chn1.value != chn3.value
                end
            end
        end

        @testset "ordering of parameters follows that of model" begin
            @model function f()
                a ~ Normal()
                x = zeros(2)
                x .~ Normal()
                return b ~ Normal()
            end

            @testset "FlexiChains" begin
                chn = sample(f(), NUTS(), 10; chain_type=VNChain, verbose=false)
                @test FlexiChains.parameters(chn) == [@varname(a), @varname(x), @varname(b)]
            end

            @testset "MCMCChains" begin
                chn = sample(f(), NUTS(), 10; chain_type=MCMCChains.Chains, verbose=false)
                chn = MCMCChains.get_sections(chn, :parameters)
                @test MCMCChains.names(chn) == Symbol.(["a", "x[1]", "x[2]", "b"])
            end
        end

        @testset "with another sampler: $spl_name" for (spl_name, spl) in
                                                       [("MH", MH()), ("HMC", HMC(0.1, 10))]
            @testset "FlexiChains" begin
                chn = sample(model, spl, 20; chain_type=VNChain, verbose=false)
                @test chn isa VNChain
                @test size(chn) == (20, 1)
            end
            @testset "MCMCChains" begin
                chn = sample(model, spl, 20; chain_type=MCMCChains.Chains, verbose=false)
                @test chn isa MCMCChains.Chains
                @test size(chn)[1] == 20
                @test size(chn)[3] == 1
            end
        end

        @testset "FlexiChains with a custom sampler" begin
            # Set up the sampler itself.
            struct S <: AbstractMCMC.AbstractSampler end
            struct Tn end
            AbstractMCMC.step(
                rng::Random.AbstractRNG,
                model::DynamicPPL.Model,
                ::S,
                state=nothing;
                kwargs...,
            ) = (Tn(), nothing)
            # Get it to work with FlexiChains
            FlexiChains.to_vnt_and_stats(::Tn) = (VarNamedTuple(; x=1), (; b="hi"))
            # Then we should be able to sample
            chn = sample(model, S(), 20; chain_type=VNChain)
            @test chn isa VNChain
            @test size(chn) == (20, 1)
            @test all(x -> x == 1, vec(chn[@varname(x)]))
            @test all(x -> x == "hi", vec(chn[Extra(:b)]))
        end
    end

    @testset "FlexiChains  metadata" begin
        @testset "sampling time exists" begin
            @model f() = x ~ Normal()
            model = f()

            @testset "single chain" begin
                chn = sample(model, NUTS(), 100; chain_type=VNChain, verbose=false)
                @test only(FlexiChains.sampling_time(chn)) isa AbstractFloat
            end
            @testset "multiple chain" begin
                chn = sample(
                    model, NUTS(), MCMCThreads(), 100, 3; chain_type=VNChain, verbose=false
                )
                @test FlexiChains.sampling_time(chn) isa AbstractVector{<:AbstractFloat}
                @test length(FlexiChains.sampling_time(chn)) == 3
            end
        end

        @testset "save_state and initial_state" begin
            @model f() = x ~ Normal()
            model = f()

            @testset "single chain" begin
                chn1 = sample(
                    model,
                    StaticSampler(),
                    10;
                    chain_type=VNChain,
                    verbose=false,
                    save_state=true,
                )
                # check that the sampler state is stored
                @test only(FlexiChains.last_sampler_state(chn1)) isa
                    DynamicPPL.VarNamedTuple
                # check that it can be resumed from
                chn2 = sample(
                    model,
                    StaticSampler(),
                    10;
                    chain_type=VNChain,
                    verbose=false,
                    initial_state=only(FlexiChains.last_sampler_state(chn1)),
                )
                # check that it did reuse the previous state
                xval = chn1[@varname(x)][end]
                @test all(x -> x == xval, chn2[@varname(x)])
            end

            @testset "multiple chain" begin
                chn1 = sample(
                    model,
                    StaticSampler(),
                    MCMCThreads(),
                    10,
                    3;
                    chain_type=VNChain,
                    verbose=false,
                    save_state=true,
                )
                # check that the sampler state is stored
                @test FlexiChains.last_sampler_state(chn1) isa
                    AbstractVector{<:DynamicPPL.VarNamedTuple}
                @test length(FlexiChains.last_sampler_state(chn1)) == 3
                # check that it can be resumed from
                chn2 = sample(
                    model,
                    StaticSampler(),
                    MCMCThreads(),
                    10,
                    3;
                    chain_type=VNChain,
                    verbose=false,
                    initial_state=FlexiChains.last_sampler_state(chn1),
                )
                # check that it did reuse the previous state
                xval = chn1[@varname(x)][end, :]
                @test all(i -> chn2[@varname(x)][i, :] == xval, 1:10)
            end
        end
    end
end

@testset "underlying data is same regardless of backend" begin
    @model function demomodel(x)
        m ~ Normal(0, 1.0)
        x ~ Normal(m, 1.0)
        return nothing
    end
    model = demomodel(1.5)

    chn_flexi = sample(Xoshiro(468), model, NUTS(), 100; chain_type=VNChain, verbose=false)
    chn_mcmc = sample(
        Xoshiro(468), model, NUTS(), 100; chain_type=MCMCChains.Chains, verbose=false
    )
    @test vec(chn_flexi[@varname(m)]) == vec(chn_mcmc[:m])
    for lp_type in [:logprior, :loglikelihood, :logjoint]
        @test vec(chn_flexi[Extra(lp_type)]) == vec(chn_mcmc[lp_type])
    end
end

end # module
