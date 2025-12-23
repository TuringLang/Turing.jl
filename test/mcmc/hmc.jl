module HMCTests

using ..Models: gdemo_default
using ..NumericalTests: check_gdemo, check_numerical
using Bijectors: Bijectors
using Distributions: Bernoulli, Beta, Categorical, Dirichlet, Normal, Wishart, sample
using DynamicPPL: DynamicPPL
import ForwardDiff
using HypothesisTests: ApproximateTwoSampleKSTest, pvalue
import ReverseDiff
using LinearAlgebra: I, dot, vec
import Random
using StableRNGs: StableRNG
using StatsFuns: logistic
using Test: @test, @test_logs, @testset, @test_throws
using Turing

@testset verbose = true "Testing hmc.jl" begin
    @info "Starting HMC tests"
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
            HMC(1.5, 3),# using a large step size (1.5)
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

        chain = sample(StableRNG(seed), constrained_simplex_test(obs12), HMC(0.75, 2), 1000)

        check_numerical(chain, ["ps[1]", "ps[2]"], [5 / 16, 11 / 16]; atol=0.015)
    end

    # Test the sampling of a matrix-value distribution.
    @testset "matrix support" begin
        dist = Wishart(7, [1 0.5; 0.5 1])
        @model hmcmatrixsup() = v ~ dist
        model_f = hmcmatrixsup()
        n_samples = 1_000

        chain = sample(StableRNG(24), model_f, HMC(0.15, 7), n_samples)
        # Reshape the chain into an array of 2x2 matrices, one per sample. Then compute
        # the average of the samples, as a matrix
        r = reshape(Array(chain), n_samples, 2, 2)
        r_mean = dropdims(mean(r; dims=1); dims=1)

        @test isapprox(r_mean, mean(dist); atol=0.2)
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

        @model function bnn(ts)
            b1 ~ MvNormal(
                [0.0; 0.0; 0.0], [var_prior 0.0 0.0; 0.0 var_prior 0.0; 0.0 0.0 var_prior]
            )
            w11 ~ MvNormal([0.0; 0.0], [var_prior 0.0; 0.0 var_prior])
            w12 ~ MvNormal([0.0; 0.0], [var_prior 0.0; 0.0 var_prior])
            w13 ~ MvNormal([0.0; 0.0], [var_prior 0.0; 0.0 var_prior])
            bo ~ Normal(0, var_prior)

            wo ~ MvNormal(
                [0.0; 0; 0], [var_prior 0.0 0.0; 0.0 var_prior 0.0; 0.0 0.0 var_prior]
            )
            for i in rand(1:N, 10)
                y = nn(xs[i], b1, w11, w12, w13, bo, wo)
                ts[i] ~ Bernoulli(y)
            end
            return b1, w11, w12, w13, bo, wo
        end

        # Sampling
        chain = sample(StableRNG(seed), bnn(ts), HMC(0.1, 5), 10)
    end

    @testset "hmcda inference" begin
        alg1 = HMCDA(500, 0.8, 0.015)
        res1 = sample(StableRNG(seed), gdemo_default, alg1, 3_000)
        check_gdemo(res1)
    end

    # TODO(mhauru) The below one is a) slow, b) flaky, in that changing the seed can
    # easily make it fail, despite many more samples than taken by most other tests. Hence
    # explicitly specifying the seeds here.
    @testset "hmcda+gibbs inference" begin
        Random.seed!(12345)
        alg = Gibbs(:s => PG(20), :m => HMCDA(500, 0.8, 0.25; init_ϵ=0.05))
        res = sample(StableRNG(123), gdemo_default, alg, 3000; discard_initial=1000)
        check_gdemo(res)
    end

    @testset "nuts inference" begin
        alg = NUTS(1000, 0.8)
        res = sample(StableRNG(seed), gdemo_default, alg, 5_000)
        check_gdemo(res)
    end

    @testset "check discard" begin
        alg = NUTS(100, 0.8)

        c1 = sample(StableRNG(seed), gdemo_default, alg, 500; discard_adapt=true)
        c2 = sample(StableRNG(seed), gdemo_default, alg, 500; discard_adapt=false)

        @test size(c1, 1) == 500
        @test size(c2, 1) == 500
    end

    @testset "AHMC resize" begin
        alg1 = Gibbs(:m => PG(10), :s => NUTS(100, 0.65))
        alg2 = Gibbs(:m => PG(10), :s => HMC(0.1, 3))
        alg3 = Gibbs(:m => PG(10), :s => HMCDA(100, 0.65, 0.3))
        @test sample(StableRNG(seed), gdemo_default, alg1, 10) isa Chains
        @test sample(StableRNG(seed), gdemo_default, alg2, 10) isa Chains
        @test sample(StableRNG(seed), gdemo_default, alg3, 10) isa Chains
    end

    # issue #1923
    @testset "reproducibility" begin
        alg = NUTS(1000, 0.8)
        res1 = sample(StableRNG(seed), gdemo_default, alg, 10)
        res2 = sample(StableRNG(seed), gdemo_default, alg, 10)
        res3 = sample(StableRNG(seed), gdemo_default, alg, 10)
        @test Array(res1) == Array(res2) == Array(res3)
    end

    @testset "initial params are respected" begin
        @model demo_norm() = x ~ Beta(2, 2)
        init_x = 0.5
        @testset "$spl_name" for (spl_name, spl) in
                                 (("HMC", HMC(0.1, 10)), ("NUTS", NUTS()))
            chain = sample(
                demo_norm(),
                spl,
                5;
                discard_adapt=false,
                initial_params=InitFromParams((x=init_x,)),
            )
            @test chain[:x][1] == init_x
            chain = sample(
                demo_norm(),
                spl,
                MCMCThreads(),
                5,
                5;
                discard_adapt=false,
                initial_params=(fill(InitFromParams((x=init_x,)), 5)),
            )
            @test all(chain[:x][1, :] .== init_x)
        end
    end


    @testset "NUTS initial parameters" begin
        @model function f()
            x ~ Normal()
            return 10 ~ Normal(x)
        end
        chn1 = sample(StableRNG(468), f(), NUTS(), 100; save_state=true)
        # chn1 should end up around x = 5.
        chn2 = sample(
            StableRNG(468),
            f(),
            NUTS(),
            10;
            nadapts=0,
            discard_adapt=false,
            initial_state=loadstate(chn1),
        )
        # if chn2 uses initial_state, its first sample should be somewhere around 5. if
        # initial_state isn't used, it will be sampled from [-2, 2] so this test should fail
        @test isapprox(chn2[:x][1], 5.0; atol=2.0)
    end

    @testset "(partially) issue: #2095" begin
        @model function vector_of_dirichlet((::Type{TV})=Vector{Float64}) where {TV}
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
                x ~ Bijectors.transformed(
                    Normal(0, 1), Bijectors.inverse(Bijectors.Logit(lb, ub))
                )
            catch e
                if e isa DomainError
                    @addlogprob! -Inf
                    return nothing
                else
                    rethrow()
                end
            end
        end

        model = buggy_model()
        num_samples = 1_000

        initial_params = InitFromParams((lb=0.5, ub=1.75, x=1.0))
        chain = sample(model, NUTS(), num_samples; initial_params=initial_params)
        chain_prior = sample(model, Prior(), num_samples)

        # Extract the `x` like this because running `generated_quantities` was how
        # the issue was discovered, hence we also want to make sure that it works.
        results = returned(model, chain)
        results_prior = returned(model, chain_prior)

        # Make sure none of the samples in the chains resulted in errors.
        @test all(!isnothing, results)

        # The discrepancies in the chains are in the tails, so we can't just compare the mean, etc.
        # KS will compare the empirical CDFs, which seems like a reasonable thing to do here.
        @test pvalue(ApproximateTwoSampleKSTest(vec(results), vec(results_prior))) > 0.001
    end

    @testset "getstepsize: Turing.jl#2400" begin
        spls = [HMC(0.1, 10), HMCDA(0.8, 0.75), NUTS(0.5), NUTS(0, 0.5)]
        @testset "$(spl)" for spl in spls
            # Construct a HMC state by taking a single step
            hmc_state = Turing.Inference.initialstep(
                Random.default_rng(),
                gdemo_default,
                spl,
                DynamicPPL.VarInfo(gdemo_default);
                initial_params=InitFromUniform(),
            )[2]
            # Check that we can obtain the current step size
            @test Turing.Inference.getstepsize(spl, hmc_state) isa Float64
        end
    end

    @testset "improved error message for initialization failures" begin
        # Model that always fails to initialize
        @model function failing_model()
            x ~ Normal()
            @addlogprob! -Inf
        end

        # Test that error message includes troubleshooting link
        @test_throws ErrorException sample(failing_model(), NUTS(), 10; progress=false)
        @test_throws "https://turinglang.org/docs/uri/initial-parameters" sample(
            failing_model(), NUTS(), 10; progress=false
        )
    end
end

end
