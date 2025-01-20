module HMCTests

using ..Models: gdemo_default
using ..ADUtils: ADTypeCheckContext
using ..NumericalTests: check_gdemo, check_numerical
import ..ADUtils
using Distributions: Bernoulli, Beta, Categorical, Dirichlet, Normal, Wishart, sample
import DynamicPPL
using DynamicPPL: Sampler
import Enzyme
import ForwardDiff
using HypothesisTests: ApproximateTwoSampleKSTest, pvalue
import ReverseDiff
using LinearAlgebra: I, dot, vec
import Random
using StableRNGs: StableRNG
using StatsFuns: logistic
import Mooncake
using Test: @test, @test_broken, @test_logs, @testset, @test_throws
using Turing

@testset "Testing hmc.jl with $adbackend" for adbackend in ADUtils.adbackends
    @info "Starting HMC tests with $adbackend"
    seed = 123

    @testset "constrained bounded" begin
        obs = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]

        @model function constrained_test(obs)
            p ~ Beta(2, 2)
            for i in 1:length(obs)
                obs[i] ~ Bernoulli(p)
            end
            return p
        end

        chain = sample(
            StableRNG(seed),
            constrained_test(obs),
            HMC(1.5, 3; adtype=adbackend),# using a large step size (1.5)
            1_000,
        )

        check_numerical(chain, [:p], [10 / 14]; atol=0.1)
    end

    @testset "constrained simplex" begin
        obs12 = [1, 2, 1, 2, 2, 2, 2, 2, 2, 2]

        @model function constrained_simplex_test(obs12)
            ps ~ Dirichlet(2, 3)
            pd ~ Dirichlet(4, 1)
            for i in 1:length(obs12)
                obs12[i] ~ Categorical(ps)
            end
            return ps
        end

        chain = sample(
            StableRNG(seed),
            constrained_simplex_test(obs12),
            HMC(0.75, 2; adtype=adbackend),
            1000,
        )

        check_numerical(chain, ["ps[1]", "ps[2]"], [5 / 16, 11 / 16]; atol=0.015)
    end

    @testset "hmc reverse diff" begin
        alg = HMC(0.1, 10; adtype=adbackend)
        res = sample(StableRNG(seed), gdemo_default, alg, 4_000)
        check_gdemo(res; rtol=0.1)
    end

    # Test the sampling of a matrix-value distribution.
    @testset "matrix support" begin
        dist = Wishart(7, [1 0.5; 0.5 1])
        @model hmcmatrixsup() = v ~ dist
        model_f = hmcmatrixsup()
        n_samples = 1_000

        chain = sample(StableRNG(24), model_f, HMC(0.15, 7; adtype=adbackend), n_samples)
        # Reshape the chain into an array of 2x2 matrices, one per sample. Then compute
        # the average of the samples, as a matrix
        r = reshape(Array(chain), n_samples, 2, 2)
        r_mean = dropdims(mean(r; dims=1); dims=1)

        # TODO(mhauru) The below remains broken for Enzyme. Need to investigate why.
        @test isapprox(r_mean, mean(dist); atol=0.2) broken = (adbackend isa AutoEnzyme)
    end

    @testset "multivariate support" begin
        # Define NN flow
        function nn(x, b1, w11, w12, w13, bo, wo)
            h = tanh.([w11 w12 w13]' * x .+ b1)
            return logistic(dot(wo, h) + bo)
        end

        # Generating training data
        N = 20
        M = N ÷ 4
        x1s = rand(M) * 5
        x2s = rand(M) * 5
        xt1s = Array([[x1s[i]; x2s[i]] for i in 1:M])
        append!(xt1s, Array([[x1s[i] - 6; x2s[i] - 6] for i in 1:M]))
        xt0s = Array([[x1s[i]; x2s[i] - 6] for i in 1:M])
        append!(xt0s, Array([[x1s[i] - 6; x2s[i]] for i in 1:M]))

        xs = [xt1s; xt0s]
        ts = [ones(M); ones(M); zeros(M); zeros(M)]

        # Define model

        alpha = 0.16                  # regularizatin term
        var_prior = sqrt(1.0 / alpha) # variance of the Gaussian prior

        @model function bnn(ts, var_prior)
            b1 ~ MvNormal(zeros(3), var_prior * I)
            w11 ~ MvNormal(zeros(2), var_prior * I)
            w12 ~ MvNormal(zeros(2), var_prior * I)
            w13 ~ MvNormal(zeros(2), var_prior * I)
            bo ~ Normal(0, var_prior)

            wo ~ MvNormal(zeros(3), var_prior * I)
            for i in rand(1:N, 10)
                y = nn(xs[i], b1, w11, w12, w13, bo, wo)
                ts[i] ~ Bernoulli(y)
            end
            return b1, w11, w12, w13, bo, wo
        end

        # Sampling
        chain = sample(
            StableRNG(seed), bnn(ts, var_prior), HMC(0.1, 5; adtype=adbackend), 10
        )
    end

    @testset "hmcda inference" begin
        alg1 = HMCDA(500, 0.8, 0.015; adtype=adbackend)
        res1 = sample(StableRNG(seed), gdemo_default, alg1, 3_000)
        check_gdemo(res1)
    end

    # TODO(mhauru) The below one is a) slow, b) flaky, in that changing the seed can
    # easily make it fail, despite many more samples than taken by most other tests. Hence
    # explicitly specifying the seeds here.
    @testset "hmcda+gibbs inference" begin
        Random.seed!(12345)
        alg = Gibbs(
            :s => PG(20), :m => HMCDA(500, 0.8, 0.25; init_ϵ=0.05, adtype=adbackend)
        )
        res = sample(StableRNG(123), gdemo_default, alg, 3000; discard_initial=1000)
        check_gdemo(res)
    end

    @testset "hmcda constructor" begin
        alg = HMCDA(0.8, 0.75; adtype=adbackend)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "HMCDA"

        alg = HMCDA(200, 0.8, 0.75; adtype=adbackend)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "HMCDA"

        alg = HMCDA(200, 0.8, 0.75, :s; adtype=adbackend)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "HMCDA"

        @test isa(alg, HMCDA)
        @test isa(sampler, Sampler{<:Turing.Hamiltonian})
    end

    @testset "nuts inference" begin
        alg = NUTS(1000, 0.8; adtype=adbackend)
        res = sample(StableRNG(seed), gdemo_default, alg, 500)
        check_gdemo(res)
    end

    @testset "nuts constructor" begin
        alg = NUTS(200, 0.65; adtype=adbackend)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "NUTS"

        alg = NUTS(0.65; adtype=adbackend)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "NUTS"

        alg = NUTS(200, 0.65, :m; adtype=adbackend)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "NUTS"
    end

    @testset "check discard" begin
        alg = NUTS(100, 0.8; adtype=adbackend)

        c1 = sample(StableRNG(seed), gdemo_default, alg, 500; discard_adapt=true)
        c2 = sample(StableRNG(seed), gdemo_default, alg, 500; discard_adapt=false)

        @test size(c1, 1) == 500
        @test size(c2, 1) == 500
    end

    @testset "AHMC resize" begin
        alg1 = Gibbs(:m => PG(10), :s => NUTS(100, 0.65; adtype=adbackend))
        alg2 = Gibbs(:m => PG(10), :s => HMC(0.1, 3; adtype=adbackend))
        alg3 = Gibbs(:m => PG(10), :s => HMCDA(100, 0.65, 0.3; adtype=adbackend))
        @test sample(StableRNG(seed), gdemo_default, alg1, 10) isa Chains
        @test sample(StableRNG(seed), gdemo_default, alg2, 10) isa Chains
        @test sample(StableRNG(seed), gdemo_default, alg3, 10) isa Chains
    end

    @testset "Regression tests" begin
        # https://github.com/TuringLang/DynamicPPL.jl/issues/27
        @model function mwe1(::Type{T}=Float64) where {T<:Real}
            m = Matrix{T}(undef, 2, 3)
            return m .~ MvNormal(zeros(2), I)
        end
        @test sample(StableRNG(seed), mwe1(), HMC(0.2, 4; adtype=adbackend), 100) isa Chains

        @model function mwe2(::Type{T}=Matrix{Float64}) where {T}
            m = T(undef, 2, 3)
            return m .~ MvNormal(zeros(2), I)
        end
        @test sample(StableRNG(seed), mwe2(), HMC(0.2, 4; adtype=adbackend), 100) isa Chains

        # https://github.com/TuringLang/Turing.jl/issues/1308
        @model function mwe3(::Type{T}=Array{Float64}) where {T}
            m = T(undef, 2, 3)
            return m .~ MvNormal(zeros(2), I)
        end
        @test sample(StableRNG(seed), mwe3(), HMC(0.2, 4; adtype=adbackend), 100) isa Chains
    end

    # issue #1923
    @testset "reproducibility" begin
        alg = NUTS(1000, 0.8; adtype=adbackend)
        res1 = sample(StableRNG(seed), gdemo_default, alg, 10)
        res2 = sample(StableRNG(seed), gdemo_default, alg, 10)
        res3 = sample(StableRNG(seed), gdemo_default, alg, 10)
        @test Array(res1) == Array(res2) == Array(res3)
    end

    @testset "prior" begin
        @model function demo_hmc_prior()
            # NOTE: Used to use `InverseGamma(2, 3)` but this has infinite variance
            # which means that it's _very_ difficult to find a good tolerance in the test below:)
            s ~ truncated(Normal(3, 1); lower=0)
            return m ~ Normal(0, sqrt(s))
        end
        alg = NUTS(1000, 0.8; adtype=adbackend)
        gdemo_default_prior = DynamicPPL.contextualize(
            demo_hmc_prior(), DynamicPPL.PriorContext()
        )
        chain = sample(gdemo_default_prior, alg, 500; initial_params=[3.0, 0.0])
        check_numerical(
            chain, [:s, :m], [mean(truncated(Normal(3, 1); lower=0)), 0]; atol=0.2
        )
    end

    @testset "warning for difficult init params" begin
        attempt = 0
        @model function demo_warn_initial_params()
            x ~ Normal()
            if (attempt += 1) < 30
                Turing.@addlogprob! -Inf
            end
        end

        @test_logs (
            :warn,
            "failed to find valid initial parameters in 10 tries; consider providing explicit initial parameters using the `initial_params` keyword",
        ) (:info,) match_mode = :any begin
            sample(demo_warn_initial_params(), NUTS(; adtype=adbackend), 5)
        end
    end

    @testset "error for impossible model" begin
        @model function demo_impossible()
            x ~ Normal()
            Turing.@addlogprob! -Inf
        end

        @test_throws ErrorException sample(demo_impossible(), NUTS(; adtype=adbackend), 5)
    end

    @testset "(partially) issue: #2095" begin
        @model function vector_of_dirichlet(::Type{TV}=Vector{Float64}) where {TV}
            xs = Vector{TV}(undef, 2)
            xs[1] ~ Dirichlet(ones(5))
            return xs[2] ~ Dirichlet(ones(5))
        end
        model = vector_of_dirichlet()
        chain = sample(model, NUTS(), 1_000)
        @test mean(Array(chain)) ≈ 0.2
    end

    @testset "issue: #2195" begin
        @model function buggy_model()
            lb ~ Uniform(0, 1)
            ub ~ Uniform(1.5, 2)

            # HACK: Necessary to avoid NUTS failing during adaptation.
            try
                x ~ transformed(Normal(0, 1), inverse(Bijectors.Logit(lb, ub)))
            catch e
                if e isa DomainError
                    Turing.@addlogprob! -Inf
                    return nothing
                else
                    rethrow()
                end
            end
        end

        model = buggy_model()
        num_samples = 1_000

        chain = sample(model, NUTS(), num_samples; initial_params=[0.5, 1.75, 1.0])
        chain_prior = sample(model, Prior(), num_samples)

        # Extract the `x` like this because running `generated_quantities` was how
        # the issue was discovered, hence we also want to make sure that it works.
        results = generated_quantities(model, chain)
        results_prior = generated_quantities(model, chain_prior)

        # Make sure none of the samples in the chains resulted in errors.
        @test all(!isnothing, results)

        # The discrepancies in the chains are in the tails, so we can't just compare the mean, etc.
        # KS will compare the empirical CDFs, which seems like a reasonable thing to do here.
        @test pvalue(ApproximateTwoSampleKSTest(vec(results), vec(results_prior))) > 0.001
    end

    @testset "getstepsize: Turing.jl#2400" begin
        algs = [HMC(0.1, 10), HMCDA(0.8, 0.75), NUTS(0.5), NUTS(0, 0.5)]
        @testset "$(alg)" for alg in algs
            # Construct a HMC state by taking a single step
            spl = Sampler(alg, gdemo_default)
            hmc_state = DynamicPPL.initialstep(
                Random.default_rng(), gdemo_default, spl, DynamicPPL.VarInfo(gdemo_default)
            )[2]
            # Check that we can obtain the current step size
            @test Turing.Inference.getstepsize(spl, hmc_state) isa Float64
        end
    end

    @testset "Check ADType" begin
        # These tests don't make sense for Enzyme, since it does not use a particular element
        # type.
        if !(adbackend isa AutoEnzyme)
            alg = HMC(0.1, 10; adtype=adbackend)
            m = DynamicPPL.contextualize(
                gdemo_default, ADTypeCheckContext(adbackend, gdemo_default.context)
            )
            # These will error if the adbackend being used is not the one set.
            sample(StableRNG(seed), m, alg, 10)
        end
    end
end

end
