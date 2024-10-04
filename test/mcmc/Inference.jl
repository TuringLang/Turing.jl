module InferenceTests

using ..Models: gdemo_d, gdemo_default
using ..NumericalTests: check_gdemo, check_numerical
import ..ADUtils
using Distributions: Bernoulli, Beta, InverseGamma, Normal
using Distributions: sample
import DynamicPPL
using DynamicPPL: Sampler, getlogp
import ForwardDiff
using LinearAlgebra: I
import MCMCChains
import Random
import ReverseDiff
import Mooncake
using Test: @test, @test_throws, @testset
using Turing

@testset "Testing inference.jl with $adbackend" for adbackend in ADUtils.adbackends
    # Only test threading if 1.3+.
    if VERSION > v"1.2"
        @testset "threaded sampling" begin
            # Test that chains with the same seed will sample identically.
            @testset "rng" begin
                model = gdemo_default

                # multithreaded sampling with PG causes segfaults on Julia 1.5.4
                # https://github.com/TuringLang/Turing.jl/issues/1571
                samplers = @static if VERSION <= v"1.5.3" || VERSION >= v"1.6.0"
                    (
                        HMC(0.1, 7; adtype=adbackend),
                        PG(10),
                        IS(),
                        MH(),
                        Gibbs(PG(3, :s), HMC(0.4, 8, :m; adtype=adbackend)),
                        Gibbs(HMC(0.1, 5, :s; adtype=adbackend), ESS(:m)),
                    )
                else
                    (
                        HMC(0.1, 7; adtype=adbackend),
                        IS(),
                        MH(),
                        Gibbs(HMC(0.1, 5, :s; adtype=adbackend), ESS(:m)),
                    )
                end
                for sampler in samplers
                    Random.seed!(5)
                    chain1 = sample(model, sampler, MCMCThreads(), 1000, 4)

                    Random.seed!(5)
                    chain2 = sample(model, sampler, MCMCThreads(), 1000, 4)

                    @test chain1.value == chain2.value
                end

                # Should also be stable with am explicit RNG
                seed = 5
                rng = Random.MersenneTwister(seed)
                for sampler in samplers
                    Random.seed!(rng, seed)
                    chain1 = sample(rng, model, sampler, MCMCThreads(), 1000, 4)

                    Random.seed!(rng, seed)
                    chain2 = sample(rng, model, sampler, MCMCThreads(), 1000, 4)

                    @test chain1.value == chain2.value
                end
            end

            # Smoke test for default sample call.
            Random.seed!(100)
            chain = sample(
                gdemo_default, HMC(0.1, 7; adtype=adbackend), MCMCThreads(), 1000, 4
            )
            check_gdemo(chain)

            # run sampler: progress logging should be disabled and
            # it should return a Chains object
            sampler = Sampler(HMC(0.1, 7; adtype=adbackend), gdemo_default)
            chains = sample(gdemo_default, sampler, MCMCThreads(), 1000, 4)
            @test chains isa MCMCChains.Chains
        end
    end
    @testset "chain save/resume" begin
        Random.seed!(1234)

        alg1 = HMCDA(1000, 0.65, 0.15; adtype=adbackend)
        alg2 = PG(20)
        alg3 = Gibbs(PG(30, :s), HMC(0.2, 4, :m; adtype=adbackend))

        chn1 = sample(gdemo_default, alg1, 5000; save_state=true)
        check_gdemo(chn1)

        chn1_contd = sample(gdemo_default, alg1, 5000; resume_from=chn1)
        check_gdemo(chn1_contd)

        chn1_contd2 = sample(gdemo_default, alg1, 5000; resume_from=chn1)
        check_gdemo(chn1_contd2)

        chn2 = sample(gdemo_default, alg2, 5000; discard_initial=2000, save_state=true)
        check_gdemo(chn2)

        chn2_contd = sample(gdemo_default, alg2, 2000; resume_from=chn2)
        check_gdemo(chn2_contd)

        chn3 = sample(gdemo_default, alg3, 5_000; discard_initial=2000, save_state=true)
        check_gdemo(chn3)

        chn3_contd = sample(gdemo_default, alg3, 5_000; resume_from=chn3)
        check_gdemo(chn3_contd)
    end
    @testset "Contexts" begin
        # Test LikelihoodContext
        @model function testmodel1(x)
            a ~ Beta()
            lp1 = getlogp(__varinfo__)
            x[1] ~ Bernoulli(a)
            return global loglike = getlogp(__varinfo__) - lp1
        end
        model = testmodel1([1.0])
        varinfo = Turing.VarInfo(model)
        model(varinfo, Turing.SampleFromPrior(), Turing.LikelihoodContext())
        @test getlogp(varinfo) == loglike

        # Test MiniBatchContext
        @model function testmodel2(x)
            a ~ Beta()
            return x[1] ~ Bernoulli(a)
        end
        model = testmodel2([1.0])
        varinfo1 = Turing.VarInfo(model)
        varinfo2 = deepcopy(varinfo1)
        model(varinfo1, Turing.SampleFromPrior(), Turing.LikelihoodContext())
        model(
            varinfo2,
            Turing.SampleFromPrior(),
            Turing.MiniBatchContext(Turing.LikelihoodContext(), 10),
        )
        @test isapprox(getlogp(varinfo2) / getlogp(varinfo1), 10)
    end
    @testset "Prior" begin
        N = 10_000

        # Note that all chains contain 3 values per sample: 2 variables + log probability
        Random.seed!(100)
        chains = sample(gdemo_d(), Prior(), N)
        @test chains isa MCMCChains.Chains
        @test size(chains) == (N, 3, 1)
        @test mean(chains, :s) ≈ 3 atol = 0.1
        @test mean(chains, :m) ≈ 0 atol = 0.1

        Random.seed!(100)
        chains = sample(gdemo_d(), Prior(), MCMCThreads(), N, 4)
        @test chains isa MCMCChains.Chains
        @test size(chains) == (N, 3, 4)
        @test mean(chains, :s) ≈ 3 atol = 0.1
        @test mean(chains, :m) ≈ 0 atol = 0.1

        Random.seed!(100)
        chains = sample(gdemo_d(), Prior(), N; chain_type=Vector{NamedTuple})
        @test chains isa Vector{<:NamedTuple}
        @test length(chains) == N
        @test all(length(x) == 3 for x in chains)
        @test all(haskey(x, :lp) for x in chains)
        @test mean(x[:s][1] for x in chains) ≈ 3 atol = 0.1
        @test mean(x[:m][1] for x in chains) ≈ 0 atol = 0.1

        @testset "#2169" begin
            # Not exactly the same as the issue, but similar.
            @model function issue2169_model()
                if DynamicPPL.leafcontext(__context__) isa DynamicPPL.PriorContext
                    x ~ Normal(0, 1)
                else
                    x ~ Normal(1000, 1)
                end
            end

            model = issue2169_model()
            chain = sample(model, Prior(), 10)
            @test all(mean(chain[:x]) .< 5)
        end
    end

    @testset "chain ordering" begin
        for alg in (Prior(), Emcee(10, 2.0))
            chain_sorted = sample(gdemo_default, alg, 1; sort_chain=true)
            @test names(MCMCChains.get_sections(chain_sorted, :parameters)) == [:m, :s]

            chain_unsorted = sample(gdemo_default, alg, 1; sort_chain=false)
            @test names(MCMCChains.get_sections(chain_unsorted, :parameters)) == [:s, :m]
        end
    end

    @testset "chain iteration numbers" begin
        for alg in (Prior(), Emcee(10, 2.0))
            chain = sample(gdemo_default, alg, 10)
            @test range(chain) == 1:10

            chain = sample(gdemo_default, alg, 10; discard_initial=5, thinning=2)
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

        res1 = sample(test_assume(), smc, 1000)
        res2 = sample(test_assume(), pg, 1000)

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
        gibbs = Gibbs(HMC(0.2, 3, :p; adtype=adbackend), PG(10, :x))

        chn_s = sample(testbb(obs), smc, 1000)
        chn_p = sample(testbb(obs), pg, 2000)
        chn_g = sample(testbb(obs), gibbs, 1500)

        check_numerical(chn_s, [:p], [meanp]; atol=0.05)
        check_numerical(chn_p, [:x], [meanp]; atol=0.1)
        check_numerical(chn_g, [:x], [meanp]; atol=0.1)
    end
    @testset "forbid global" begin
        xs = [1.5 2.0]
        # xx = 1

        @model function fggibbstest(xs)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            # xx ~ Normal(m, sqrt(s)) # this is illegal

            for i in 1:length(xs)
                xs[i] ~ Normal(m, sqrt(s))
                # for xx in xs
                # xx ~ Normal(m, sqrt(s))
            end
            return s, m
        end

        gibbs = Gibbs(PG(10, :s), HMC(0.4, 8, :m; adtype=adbackend))
        chain = sample(fggibbstest(xs), gibbs, 2)
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

        chain = sample(gauss(x), PG(10), 10)
        chain = sample(gauss(x), SMC(), 10)

        @model function gauss2(::Type{TV}=Vector{Float64}; x) where {TV}
            priors = TV(undef, 2)
            priors[1] ~ InverseGamma(2, 3)         # s
            priors[2] ~ Normal(0, sqrt(priors[1])) # m
            for i in 1:length(x)
                x[i] ~ Normal(priors[2], sqrt(priors[1]))
            end
            return priors
        end

        @test_throws ErrorException chain = sample(gauss2(; x=x), PG(10), 10)
        @test_throws ErrorException chain = sample(gauss2(; x=x), SMC(), 10)

        @test_throws ErrorException chain = sample(
            gauss2(DynamicPPL.TypeWrap{Vector{Float64}}(); x=x), PG(10), 10
        )
        @test_throws ErrorException chain = sample(
            gauss2(DynamicPPL.TypeWrap{Vector{Float64}}(); x=x), SMC(), 10
        )

        @model function gauss3(x, ::Type{TV}=Vector{Float64}) where {TV}
            priors = TV(undef, 2)
            priors[1] ~ InverseGamma(2, 3)         # s
            priors[2] ~ Normal(0, sqrt(priors[1])) # m
            for i in 1:length(x)
                x[i] ~ Normal(priors[2], sqrt(priors[1]))
            end
            return priors
        end

        chain = sample(gauss3(x), PG(10), 10)
        chain = sample(gauss3(x), SMC(), 10)

        chain = sample(gauss3(x, DynamicPPL.TypeWrap{Vector{Real}}()), PG(10), 10)
        chain = sample(gauss3(x, DynamicPPL.TypeWrap{Vector{Real}}()), SMC(), 10)
    end
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
            newinterface(obs),
            HMC(0.75, 3, :p, :x; adtype=Turing.AutoForwardDiff(; chunksize=2)),
            100,
        )
    end
    @testset "no return" begin
        Random.seed!(5)
        @model function noreturn(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            for i in 1:length(x)
                x[i] ~ Normal(m, sqrt(s))
            end
        end

        chain = sample(noreturn([1.5 2.0]), HMC(0.1, 10; adtype=adbackend), 4000)
        check_numerical(chain, [:s, :m], [49 / 24, 7 / 6])
    end
    @testset "observe" begin
        Random.seed!(5)
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

        res_is = sample(test(), is, 10000)
        res_smc = sample(test(), smc, 1000)
        res_pg = sample(test(), pg, 100)

        @test all(isone, res_is[:x])
        @test res_is.logevidence ≈ 2 * log(0.5)

        @test all(isone, res_smc[:x])
        @test res_smc.logevidence ≈ 2 * log(0.5)

        @test all(isone, res_pg[:x])
    end
    @testset "sample" begin
        alg = Gibbs(HMC(0.2, 3, :m; adtype=adbackend), PG(10, :s))
        chn = sample(gdemo_default, alg, 1000)
    end
    @testset "vectorization @." begin
        # https://github.com/FluxML/Tracker.jl/issues/119
        @model function vdemo1(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            @. x ~ Normal(m, sqrt(s))
            return s, m
        end

        alg = HMC(0.01, 5; adtype=adbackend)
        x = randn(100)
        res = sample(vdemo1(x), alg, 250)

        @model function vdemo1b(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            @. x ~ Normal(m, $(sqrt(s)))
            return s, m
        end

        res = sample(vdemo1b(x), alg, 250)

        @model function vdemo2(x)
            μ ~ MvNormal(zeros(size(x, 1)), I)
            @. x ~ $(MvNormal(μ, I))
        end

        D = 2
        alg = HMC(0.01, 5; adtype=adbackend)
        res = sample(vdemo2(randn(D, 100)), alg, 250)

        # Vector assumptions
        N = 10
        alg = HMC(0.2, 4; adtype=adbackend)

        @model function vdemo3()
            x = Vector{Real}(undef, N)
            for i in 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        t_loop = @elapsed res = sample(vdemo3(), alg, 1000)

        # Test for vectorize UnivariateDistribution
        @model function vdemo4()
            x = Vector{Real}(undef, N)
            @. x ~ Normal(0, 2)
        end

        t_vec = @elapsed res = sample(vdemo4(), alg, 1000)

        @model vdemo5() = x ~ MvNormal(zeros(N), 4 * I)

        t_mv = @elapsed res = sample(vdemo5(), alg, 1000)

        println("Time for")
        println("  Loop : ", t_loop)
        println("  Vec  : ", t_vec)
        println("  Mv   : ", t_mv)

        # Transformed test
        @model function vdemo6()
            x = Vector{Real}(undef, N)
            @. x ~ InverseGamma(2, 3)
        end

        sample(vdemo6(), alg, 1000)

        N = 3
        @model function vdemo7()
            x = Array{Real}(undef, N, N)
            @. x ~ [InverseGamma(2, 3) for i in 1:N]
        end

        sample(vdemo7(), alg, 1000)
    end
    @testset "vectorization .~" begin
        @model function vdemo1(x)
            s ~ InverseGamma(2, 3)
            m ~ Normal(0, sqrt(s))
            x .~ Normal(m, sqrt(s))
            return s, m
        end

        alg = HMC(0.01, 5; adtype=adbackend)
        x = randn(100)
        res = sample(vdemo1(x), alg, 250)

        @model function vdemo2(x)
            μ ~ MvNormal(zeros(size(x, 1)), I)
            return x .~ MvNormal(μ, I)
        end

        D = 2
        alg = HMC(0.01, 5; adtype=adbackend)
        res = sample(vdemo2(randn(D, 100)), alg, 250)

        # Vector assumptions
        N = 10
        alg = HMC(0.2, 4; adtype=adbackend)

        @model function vdemo3()
            x = Vector{Real}(undef, N)
            for i in 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        t_loop = @elapsed res = sample(vdemo3(), alg, 1000)

        # Test for vectorize UnivariateDistribution
        @model function vdemo4()
            x = Vector{Real}(undef, N)
            return x .~ Normal(0, 2)
        end

        t_vec = @elapsed res = sample(vdemo4(), alg, 1000)

        @model vdemo5() = x ~ MvNormal(zeros(N), 4 * I)

        t_mv = @elapsed res = sample(vdemo5(), alg, 1000)

        println("Time for")
        println("  Loop : ", t_loop)
        println("  Vec  : ", t_vec)
        println("  Mv   : ", t_mv)

        # Transformed test
        @model function vdemo6()
            x = Vector{Real}(undef, N)
            return x .~ InverseGamma(2, 3)
        end

        sample(vdemo6(), alg, 1000)

        @model function vdemo7()
            x = Array{Real}(undef, N, N)
            return x .~ [InverseGamma(2, 3) for i in 1:N]
        end

        sample(vdemo7(), alg, 1000)
    end
    @testset "Type parameters" begin
        N = 10
        alg = HMC(0.01, 5; adtype=adbackend)
        x = randn(1000)
        @model function vdemo1(::Type{T}=Float64) where {T}
            x = Vector{T}(undef, N)
            for i in 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        t_loop = @elapsed res = sample(vdemo1(), alg, 250)
        t_loop = @elapsed res = sample(vdemo1(DynamicPPL.TypeWrap{Float64}()), alg, 250)

        vdemo1kw(; T) = vdemo1(T)
        t_loop = @elapsed res = sample(
            vdemo1kw(; T=DynamicPPL.TypeWrap{Float64}()), alg, 250
        )

        @model function vdemo2(::Type{T}=Float64) where {T<:Real}
            x = Vector{T}(undef, N)
            @. x ~ Normal(0, 2)
        end

        t_vec = @elapsed res = sample(vdemo2(), alg, 250)
        t_vec = @elapsed res = sample(vdemo2(DynamicPPL.TypeWrap{Float64}()), alg, 250)

        vdemo2kw(; T) = vdemo2(T)
        t_vec = @elapsed res = sample(
            vdemo2kw(; T=DynamicPPL.TypeWrap{Float64}()), alg, 250
        )

        @model function vdemo3(::Type{TV}=Vector{Float64}) where {TV<:AbstractVector}
            x = TV(undef, N)
            @. x ~ InverseGamma(2, 3)
        end

        sample(vdemo3(), alg, 250)
        sample(vdemo3(DynamicPPL.TypeWrap{Vector{Float64}}()), alg, 250)

        vdemo3kw(; T) = vdemo3(T)
        sample(vdemo3kw(; T=DynamicPPL.TypeWrap{Vector{Float64}}()), alg, 250)
    end

    @testset "names_values" begin
        ks, xs = Turing.Inference.names_values([(a=1,), (b=2,), (a=3, b=4)])
        @test all(xs[:, 1] .=== [1, missing, 3])
        @test all(xs[:, 2] .=== [missing, 2, 4])
    end

    @testset "check model" begin
        @model function demo_repeated_varname()
            x ~ Normal(0, 1)
            return x ~ Normal(x, 1)
        end

        @test_throws ErrorException sample(
            demo_repeated_varname(), NUTS(), 1000; check_model=true
        )
        # Make sure that disabling the check also works.
        @test (sample(demo_repeated_varname(), Prior(), 10; check_model=false);
        true)

        @model function demo_incorrect_missing(y)
            return y[1:1] ~ MvNormal(zeros(1), I)
        end
        @test_throws ErrorException sample(
            demo_incorrect_missing([missing]), NUTS(), 1000; check_model=true
        )
    end
end

end
