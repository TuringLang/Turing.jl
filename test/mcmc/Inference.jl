module InferenceTests

using ..Models: gdemo_d, gdemo_default
using ..NumericalTests: check_gdemo, check_numerical
using Distributions: Bernoulli, Beta, InverseGamma, Normal
using Distributions: sample
using AbstractMCMC: AbstractMCMC
import DynamicPPL
using DynamicPPL: Sampler
import ForwardDiff
using LinearAlgebra: I
import MCMCChains
import Random
import ReverseDiff
using StableRNGs: StableRNG
using Test: @test, @test_throws, @testset
using Turing

@testset verbose = true "Testing Inference.jl" begin
    @info "Starting Inference.jl tests"

    seed = 23

    @testset "threaded sampling" begin
        # Test that chains with the same seed will sample identically.
        @testset "rng" begin
            model = gdemo_default

            samplers = (
                HMC(0.1, 7),
                PG(10),
                IS(),
                MH(),
                Gibbs(:s => PG(3), :m => HMC(0.4, 8)),
                Gibbs(:s => HMC(0.1, 5), :m => ESS()),
            )
            for sampler in samplers
                Random.seed!(5)
                chain1 = sample(model, sampler, MCMCThreads(), 10, 4)

                Random.seed!(5)
                chain2 = sample(model, sampler, MCMCThreads(), 10, 4)

                # For HMC, the first step does not have stats, so we need to use isequal to
                # avoid comparing `missing`s
                @test isequal(chain1.value, chain2.value)
            end

            # Should also be stable with an explicit RNG
            local_seed = 5
            rng = Random.MersenneTwister(local_seed)
            for sampler in samplers
                Random.seed!(rng, local_seed)
                chain1 = sample(rng, model, sampler, MCMCThreads(), 10, 4)

                Random.seed!(rng, local_seed)
                chain2 = sample(rng, model, sampler, MCMCThreads(), 10, 4)

                @test isequal(chain1.value, chain2.value)
            end
        end

        # Smoke test for default sample call.
        @testset "gdemo_default" begin
            chain = sample(
                StableRNG(seed), gdemo_default, HMC(0.1, 7), MCMCThreads(), 1_000, 4
            )
            check_gdemo(chain)

            # run sampler: progress logging should be disabled and
            # it should return a Chains object
            sampler = Sampler(HMC(0.1, 7))
            chains = sample(StableRNG(seed), gdemo_default, sampler, MCMCThreads(), 10, 4)
            @test chains isa MCMCChains.Chains
        end
    end

    @testset "save/resume correctly reloads state" begin
        struct StaticSampler <: Turing.Inference.InferenceAlgorithm end
        function DynamicPPL.initialstep(
            rng, model, ::DynamicPPL.Sampler{<:StaticSampler}, vi; kwargs...
        )
            return Turing.Inference.Transition(model, vi, nothing), vi
        end
        function AbstractMCMC.step(
            rng,
            model,
            ::DynamicPPL.Sampler{<:StaticSampler},
            vi::DynamicPPL.AbstractVarInfo;
            kwargs...,
        )
            return Turing.Inference.Transition(model, vi, nothing), vi
        end

        @model demo() = x ~ Normal()

        @testset "single-chain" begin
            chn1 = sample(demo(), StaticSampler(), 10; save_state=true)
            @test chn1.info.samplerstate isa DynamicPPL.AbstractVarInfo
            chn2 = sample(demo(), StaticSampler(), 10; resume_from=chn1)
            xval = chn1[:x][1]
            @test all(chn2[:x] .== xval)
        end

        @testset "multiple-chain" for nchains in [1, 3]
            chn1 = sample(
                demo(), StaticSampler(), MCMCThreads(), 10, nchains; save_state=true
            )
            @test chn1.info.samplerstate isa AbstractVector{<:DynamicPPL.AbstractVarInfo}
            @test length(chn1.info.samplerstate) == nchains
            chn2 = sample(
                demo(), StaticSampler(), MCMCThreads(), 10, nchains; resume_from=chn1
            )
            xval = chn1[:x][1, :]
            @test all(i -> chn2[:x][i, :] == xval, 1:10)
        end
    end

    @testset "single-chain save/resume numerical accuracy" begin
        alg1 = HMCDA(1000, 0.65, 0.15)
        alg2 = PG(20)
        alg3 = Gibbs(:s => PG(30), :m => HMC(0.2, 4))

        chn1 = sample(StableRNG(seed), gdemo_default, alg1, 10_000; save_state=true)
        check_gdemo(chn1)

        chn1_contd = sample(StableRNG(seed), gdemo_default, alg1, 2_000; resume_from=chn1)
        check_gdemo(chn1_contd)

        chn1_contd2 = sample(StableRNG(seed), gdemo_default, alg1, 2_000; resume_from=chn1)
        check_gdemo(chn1_contd2)

        chn2 = sample(
            StableRNG(seed),
            gdemo_default,
            alg2,
            2_000;
            discard_initial=100,
            save_state=true,
        )
        check_gdemo(chn2)

        chn2_contd = sample(StableRNG(seed), gdemo_default, alg2, 2_000; resume_from=chn2)
        check_gdemo(chn2_contd)

        chn3 = sample(
            StableRNG(seed),
            gdemo_default,
            alg3,
            2_000;
            discard_initial=100,
            save_state=true,
        )
        check_gdemo(chn3)

        chn3_contd = sample(StableRNG(seed), gdemo_default, alg3, 5_000; resume_from=chn3)
        check_gdemo(chn3_contd)
    end

    @testset "Prior" begin
        N = 10_000

        @testset "Single-threaded vanilla" begin
            chains = sample(StableRNG(seed), gdemo_d(), Prior(), N)
            @test chains isa MCMCChains.Chains
            @test mean(chains, :s) ≈ 3 atol = 0.11
            @test mean(chains, :m) ≈ 0 atol = 0.1
        end

        @testset "Multi-threaded" begin
            chains = sample(StableRNG(seed), gdemo_d(), Prior(), MCMCThreads(), N, 4)
            @test chains isa MCMCChains.Chains
            @test mean(chains, :s) ≈ 3 atol = 0.11
            @test mean(chains, :m) ≈ 0 atol = 0.1
        end

        @testset "Vector chain_type" begin
            chains = sample(
                StableRNG(seed), gdemo_d(), Prior(), N; chain_type=Vector{NamedTuple}
            )
            @test chains isa Vector{<:NamedTuple}
            @test length(chains) == N
            @test all(haskey(x, :lp) for x in chains)
            @test all(haskey(x, :logprior) for x in chains)
            @test all(haskey(x, :loglikelihood) for x in chains)
            @test mean(x[:s][1] for x in chains) ≈ 3 atol = 0.11
            @test mean(x[:m][1] for x in chains) ≈ 0 atol = 0.1
        end

        @testset "accumulators are set correctly" begin
            # Prior() uses `reevaluate=false` when constructing a
            # `Turing.Inference.Transition`, so we had better make sure that it
            # does capture colon-eq statements, as we can't rely on the default
            # `Transition` constructor to do this for us.
            @model function coloneq()
                x ~ Normal()
                10.0 ~ Normal(x)
                z := 1.0
                return nothing
            end
            chain = sample(coloneq(), Prior(), N)
            @test chain isa MCMCChains.Chains
            @test all(x -> x == 1.0, chain[:z])
            # And for the same reason we should also make sure that the logp
            # components are correctly calculated.
            @test isapprox(chain[:logprior], logpdf.(Normal(), chain[:x]))
            @test isapprox(chain[:loglikelihood], logpdf.(Normal.(chain[:x]), 10.0))
            @test isapprox(chain[:lp], chain[:logprior] .+ chain[:loglikelihood])
            # And that the outcome is not influenced by the likelihood
            @test mean(chain, :x) ≈ 0.0 atol = 0.1
        end
    end

    @testset "chain ordering" begin
        for alg in (Prior(), Emcee(10, 2.0))
            chain_sorted = sample(StableRNG(seed), gdemo_default, alg, 1; sort_chain=true)
            @test names(MCMCChains.get_sections(chain_sorted, :parameters)) == [:m, :s]

            chain_unsorted = sample(
                StableRNG(seed), gdemo_default, alg, 1; sort_chain=false
            )
            @test names(MCMCChains.get_sections(chain_unsorted, :parameters)) == [:s, :m]
        end
    end

    @testset "chain iteration numbers" begin
        for alg in (Prior(), Emcee(10, 2.0))
            chain = sample(StableRNG(seed), gdemo_default, alg, 10)
            @test range(chain) == 1:10

            chain = sample(
                StableRNG(seed), gdemo_default, alg, 10; discard_initial=5, thinning=2
            )
            @test range(chain) == range(6; step=2, length=10)
        end
    end

    # Copy-paste from integration tests in DynamicPPL.
    @testset "assume" begin
        @model function test_assume()
            x ~ Bernoulli(1)
            y ~ Bernoulli(x / 2)
            return x, y
        end

        smc = SMC()
        pg = PG(10)

        res1 = sample(StableRNG(seed), test_assume(), smc, 1_000)
        res2 = sample(StableRNG(seed), test_assume(), pg, 1_000)

        check_numerical(res1, [:y], [0.5]; atol=0.1)
        check_numerical(res2, [:y], [0.5]; atol=0.1)

        # Check that all xs are 1.
        @test all(isone, res1[:x])
        @test all(isone, res2[:x])
    end

    @testset "beta binomial" begin
        prior = Beta(2, 2)
        obs = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        exact = Beta(prior.α + sum(obs), prior.β + length(obs) - sum(obs))
        meanp = exact.α / (exact.α + exact.β)

        @model function testbb(obs)
            p ~ Beta(2, 2)
            x ~ Bernoulli(p)
            for i in 1:length(obs)
                obs[i] ~ Bernoulli(p)
            end
            return p, x
        end

        smc = SMC()
        pg = PG(10)
        gibbs = Gibbs(:p => HMC(0.2, 3), :x => PG(10))

        chn_s = sample(StableRNG(seed), testbb(obs), smc, 2_000)
        chn_p = sample(StableRNG(seed), testbb(obs), pg, 2_000)
        chn_g = sample(StableRNG(seed), testbb(obs), gibbs, 2_000)

        check_numerical(chn_s, [:p], [meanp]; atol=0.05)
        check_numerical(chn_p, [:x], [meanp]; atol=0.1)
        check_numerical(chn_g, [:x], [meanp]; atol=0.1)
    end

    @testset "forbid global" begin
        xs = [1.5 2.0]

        @model function fggibbstest(xs)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            for i in 1:length(xs)
                xs[i] ~ Normal(m, sqrt(s))
            end
            return s, m
        end

        gibbs = Gibbs(:s => PG(10), :m => HMC(0.4, 8))
        chain = sample(StableRNG(seed), fggibbstest(xs), gibbs, 2)
    end

    @testset "new grammar" begin
        x = Float64[1 2]

        @model function gauss(x)
            priors = zeros(Float64, 2)
            priors[1] ~ InverseGamma(2, 3)         # s
            priors[2] ~ Normal(0, sqrt(priors[1])) # m
            for i in 1:length(x)
                x[i] ~ Normal(priors[2], sqrt(priors[1]))
            end
            return priors
        end

        chain = sample(StableRNG(seed), gauss(x), PG(10), 10)
        chain = sample(StableRNG(seed), gauss(x), SMC(), 10)

        @model function gauss2((::Type{TV})=Vector{Float64}; x) where {TV}
            priors = TV(undef, 2)
            priors[1] ~ InverseGamma(2, 3)         # s
            priors[2] ~ Normal(0, sqrt(priors[1])) # m
            for i in 1:length(x)
                x[i] ~ Normal(priors[2], sqrt(priors[1]))
            end
            return priors
        end

        @test_throws ErrorException chain = sample(
            StableRNG(seed), gauss2(; x=x), PG(10), 10
        )
        @test_throws ErrorException chain = sample(
            StableRNG(seed), gauss2(; x=x), SMC(), 10
        )

        @test_throws ErrorException chain = sample(
            StableRNG(seed), gauss2(DynamicPPL.TypeWrap{Vector{Float64}}(); x=x), PG(10), 10
        )
        @test_throws ErrorException chain = sample(
            StableRNG(seed), gauss2(DynamicPPL.TypeWrap{Vector{Float64}}(); x=x), SMC(), 10
        )

        @model function gauss3(x, (::Type{TV})=Vector{Float64}) where {TV}
            priors = TV(undef, 2)
            priors[1] ~ InverseGamma(2, 3)         # s
            priors[2] ~ Normal(0, sqrt(priors[1])) # m
            for i in 1:length(x)
                x[i] ~ Normal(priors[2], sqrt(priors[1]))
            end
            return priors
        end

        chain = sample(StableRNG(seed), gauss3(x), PG(10), 10)
        chain = sample(StableRNG(seed), gauss3(x), SMC(), 10)

        chain = sample(
            StableRNG(seed), gauss3(x, DynamicPPL.TypeWrap{Vector{Real}}()), PG(10), 10
        )
        chain = sample(
            StableRNG(seed), gauss3(x, DynamicPPL.TypeWrap{Vector{Real}}()), SMC(), 10
        )
    end

    # TODO(mhauru) What is this testing? Why does it use a different adbackend?
    @testset "new interface" begin
        obs = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]

        @model function newinterface(obs)
            p ~ Beta(2, 2)
            for i in 1:length(obs)
                obs[i] ~ Bernoulli(p)
            end
            return p
        end

        sample(
            StableRNG(seed),
            newinterface(obs),
            HMC(0.75, 3; adtype=Turing.AutoForwardDiff(; chunksize=2)),
            100,
        )
    end

    @testset "no return" begin
        @model function noreturn(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            for i in 1:length(x)
                x[i] ~ Normal(m, sqrt(s))
            end
        end

        chain = sample(StableRNG(seed), noreturn([1.5 2.0]), HMC(0.1, 10), 4000)
        check_numerical(chain, [:s, :m], [49 / 24, 7 / 6])
    end

    @testset "observe" begin
        @model function test()
            z ~ Normal(0, 1)
            x ~ Bernoulli(1)
            1 ~ Bernoulli(x / 2)
            0 ~ Bernoulli(x / 2)
            return x
        end

        is = IS()
        smc = SMC()
        pg = PG(10)

        res_is = sample(StableRNG(seed), test(), is, 1_000)
        res_smc = sample(StableRNG(seed), test(), smc, 1_000)
        res_pg = sample(StableRNG(seed), test(), pg, 100)

        @test all(isone, res_is[:x])
        @test res_is.logevidence ≈ 2 * log(0.5)

        @test all(isone, res_smc[:x])
        @test res_smc.logevidence ≈ 2 * log(0.5)

        @test all(isone, res_pg[:x])
    end

    @testset "sample" begin
        alg = Gibbs(:m => HMC(0.2, 3), :s => PG(10))
        chn = sample(StableRNG(seed), gdemo_default, alg, 10)
    end

    @testset "vectorization @." begin
        @model function vdemo1(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            @. x ~ Normal(m, sqrt(s))
            return s, m
        end

        alg = HMC(0.01, 5)
        x = randn(100)
        res = sample(StableRNG(seed), vdemo1(x), alg, 10)

        @model function vdemo1b(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            @. x ~ Normal(m, $(sqrt(s)))
            return s, m
        end

        res = sample(StableRNG(seed), vdemo1b(x), alg, 10)

        # Vector assumptions
        N = 10
        alg = HMC(0.2, 4)

        @model function vdemo3()
            x = Vector{Real}(undef, N)
            for i in 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        # TODO(mhauru) What is the point of the below @elapsed stuff? It prints out some
        # timings. Do we actually ever look at them?
        t_loop = @elapsed res = sample(StableRNG(seed), vdemo3(), alg, 1000)

        # Test for vectorize UnivariateDistribution
        @model function vdemo4()
            x = Vector{Real}(undef, N)
            @. x ~ Normal(0, 2)
        end

        t_vec = @elapsed res = sample(StableRNG(seed), vdemo4(), alg, 1000)

        @model vdemo5() = x ~ MvNormal(zeros(N), 4 * I)

        t_mv = @elapsed res = sample(StableRNG(seed), vdemo5(), alg, 1000)

        println("Time for")
        println("  Loop : ", t_loop)
        println("  Vec  : ", t_vec)
        println("  Mv   : ", t_mv)

        # Transformed test
        @model function vdemo6()
            x = Vector{Real}(undef, N)
            @. x ~ InverseGamma(2, 3)
        end

        sample(StableRNG(seed), vdemo6(), alg, 10)

        N = 3
        @model function vdemo7()
            x = Array{Real}(undef, N, N)
            return x ~ filldist(InverseGamma(2, 3), N, N)
        end

        sample(StableRNG(seed), vdemo7(), alg, 10)
    end

    @testset "vectorization .~" begin
        @model function vdemo1(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            x .~ Normal(m, sqrt(s))
            return s, m
        end

        alg = HMC(0.01, 5)
        x = randn(100)
        res = sample(StableRNG(seed), vdemo1(x), alg, 10)

        @model function vdemo2(x)
            μ ~ MvNormal(zeros(size(x, 1)), I)
            return x ~ filldist(MvNormal(μ, I), size(x, 2))
        end

        D = 2
        alg = HMC(0.01, 5)
        res = sample(StableRNG(seed), vdemo2(randn(D, 100)), alg, 10)

        # Vector assumptions
        N = 10
        alg = HMC(0.2, 4)

        @model function vdemo3()
            x = Vector{Real}(undef, N)
            for i in 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        # TODO(mhauru) Same question as above about @elapsed.
        t_loop = @elapsed res = sample(StableRNG(seed), vdemo3(), alg, 1_000)

        # Test for vectorize UnivariateDistribution
        @model function vdemo4()
            x = Vector{Real}(undef, N)
            return x .~ Normal(0, 2)
        end

        t_vec = @elapsed res = sample(StableRNG(seed), vdemo4(), alg, 1_000)

        @model vdemo5() = x ~ MvNormal(zeros(N), 4 * I)

        t_mv = @elapsed res = sample(StableRNG(seed), vdemo5(), alg, 1_000)

        println("Time for")
        println("  Loop : ", t_loop)
        println("  Vec  : ", t_vec)
        println("  Mv   : ", t_mv)

        # Transformed test
        @model function vdemo6()
            x = Vector{Real}(undef, N)
            return x .~ InverseGamma(2, 3)
        end

        sample(StableRNG(seed), vdemo6(), alg, 10)

        @model function vdemo7()
            x = Array{Real}(undef, N, N)
            return x ~ filldist(InverseGamma(2, 3), N, N)
        end

        sample(StableRNG(seed), vdemo7(), alg, 10)
    end

    @testset "Type parameters" begin
        N = 10
        alg = HMC(0.01, 5)
        x = randn(1000)
        @model function vdemo1((::Type{T})=Float64) where {T}
            x = Vector{T}(undef, N)
            for i in 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        # TODO(mhauru) What are we testing below? Just that using a type parameter doesn't
        # crash?
        sample(StableRNG(seed), vdemo1(), alg, 10)
        sample(StableRNG(seed), vdemo1(DynamicPPL.TypeWrap{Float64}()), alg, 10)

        vdemo1kw(; T) = vdemo1(T)
        sample(StableRNG(seed), vdemo1kw(; T=DynamicPPL.TypeWrap{Float64}()), alg, 10)

        @model function vdemo2((::Type{T})=Float64) where {T<:Real}
            x = Vector{T}(undef, N)
            @. x ~ Normal(0, 2)
        end

        sample(StableRNG(seed), vdemo2(), alg, 10)
        sample(StableRNG(seed), vdemo2(DynamicPPL.TypeWrap{Float64}()), alg, 10)

        vdemo2kw(; T) = vdemo2(T)
        sample(StableRNG(seed), vdemo2kw(; T=DynamicPPL.TypeWrap{Float64}()), alg, 10)

        @model function vdemo3((::Type{TV})=Vector{Float64}) where {TV<:AbstractVector}
            x = TV(undef, N)
            @. x ~ InverseGamma(2, 3)
        end

        sample(StableRNG(seed), vdemo3(), alg, 10)
        sample(StableRNG(seed), vdemo3(DynamicPPL.TypeWrap{Vector{Float64}}()), alg, 10)

        vdemo3kw(; T) = vdemo3(T)
        sample(
            StableRNG(seed), vdemo3kw(; T=DynamicPPL.TypeWrap{Vector{Float64}}()), alg, 10
        )
    end

    @testset "names_values" begin
        ks, xs = Turing.Inference.names_values([(a=1,), (b=2,), (a=3, b=4)])
        @test isequal(xs[:, 1], [1, missing, 3])
        @test isequal(xs[:, 2], [missing, 2, 4])
    end

    @testset "check model" begin
        @model function demo_repeated_varname()
            x ~ Normal(0, 1)
            return x ~ Normal(x, 1)
        end

        @test_throws ErrorException sample(
            StableRNG(seed), demo_repeated_varname(), NUTS(), 10; check_model=true
        )
        # Make sure that disabling the check also works.
        @test (
            sample(
                StableRNG(seed), demo_repeated_varname(), Prior(), 10; check_model=false
            );
            true
        )

        @model function demo_incorrect_missing(y)
            return y[1:1] ~ MvNormal(zeros(1), I)
        end
        @test_throws ErrorException sample(
            StableRNG(seed), demo_incorrect_missing([missing]), NUTS(), 10; check_model=true
        )
    end

    @testset "getparams" begin
        @model function e(x=1.0)
            return x ~ Normal()
        end
        evi = DynamicPPL.VarInfo(e())
        @test isempty(Turing.Inference.getparams(e(), evi))

        @model function f()
            return x ~ Normal()
        end
        fvi = DynamicPPL.VarInfo(f())
        fparams = Turing.Inference.getparams(f(), fvi)
        @test fparams[@varname(x)] == fvi[@varname(x)]
        @test length(fparams) == 1

        @model function g()
            x ~ Normal()
            return y ~ Poisson()
        end
        gvi = DynamicPPL.VarInfo(g())
        gparams = Turing.Inference.getparams(g(), gvi)
        @test gparams[@varname(x)] == gvi[@varname(x)]
        @test gparams[@varname(y)] == gvi[@varname(y)]
        @test length(gparams) == 2
    end

    @testset "empty model" begin
        @model function e(x=1.0)
            return x ~ Normal()
        end
        # Can't test with HMC/NUTS because some AD backends error; see
        # https://github.com/JuliaDiff/DifferentiationInterface.jl/issues/802
        @test sample(e(), IS(), 100) isa MCMCChains.Chains
    end
end

end
