using Plots: histogram, plot, plot!, display
using Turing: AIS, gen_logjoint, gen_logprior, gen_log_unnorm_tempered, prior_step, intermediate_step

using AdvancedMH: RandomWalkProposal
using AbstractMCMC: sample_init!, step!, sample_end!
using DynamicPPL 

# part of the header for all tests in inference
dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@turing_testset "ais.jl" begin

    # 1. declare models and associated samplers

    # model_1: normal distributions, but multidimensional
    @model model_1_macro(x, y) = begin
        # latent
        z = Vector{Real}(undef, 2)
        z[1] ~ Normal()
        z[2] ~ Normal()
        # observed
        x[1] ~ Normal(z[1], 1.)
        x[2] ~ Normal(z[2], 1.)
        y ~ Normal(z[1] + z[2], 1.)
    end
    model_1 = model_1_macro([1., -1.], 2.)

    # declare algorithm and sampler for model_1
    schedule_1 = 0.1:0.1:0.9
    proposal_kernels_1 = [RandomWalkProposal(MvNormal(3, 1.)) for i in 1:9]
    alg_1 = AIS(proposal_kernels_1, schedule_1)
    spl_1 = Sampler(alg_1, model_1)

    # model_2: non-normal distributions, but unidimensional
    @model model_2_macro(x) = begin
        # latent
        inv_theta ~ Gamma(2,3)
        theta = 1/inv_theta
        # observed
        x ~ Weibull(1,theta) 
    end
    model_2 = model_2_macro(5.)

    # declare algorithm and sampler for model_2
    schedule_2 = 0.1:0.1:0.9
    proposal_kernels_2 = [RandomWalkProposal(MvNormal(1, 1.)) for i in 1:9]
    alg_2 = AIS(proposal_kernels_2, schedule_2)
    spl_2 = Sampler(alg_2, model_2)

    # 2. test lower level functions: gen_logjoint, gen_logprior, gen_log_unnorm_tempered

    @turing_testset "gen_logjoint" begin
        @testset "model_1" begin
            logjoint_1 = gen_logjoint(spl_1.state.vi, model_1, spl_1)
            # @test logjoint_1([???]) == ??? # TODO: set - maybe use check_numerical
            # @test logjoint_1([???]) == ??? # TODO: set - maybe use check_numerical
            # @test logjoint_1([???]) == ??? # TODO: set - maybe use check_numerical
        end

        @testset "model_2" begin
            logjoint_2 = gen_logjoint(spl_2.state.vi, model_2, spl_2)
            # @test logjoint_2([???]) == ??? # TODO: set - maybe use check_numerical
            # @test logjoint_2([???]) == ??? # TODO: set - maybe use check_numerical
            # @test logjoint_2([???]) == ??? # TODO: set - maybe use check_numerical
        end
    end

    @turing_testset "gen_logprior" begin
        @testset "model_1" begin
            logprior_1 = gen_logprior(spl_1.state.vi, model_1, spl_1)
            # @test logprior_1([???]) == ??? # TODO: set - maybe use check_numerical
            # @test logprior_1([???]) == ??? # TODO: set - maybe use check_numerical
            # @test logprior_1([???]) == ??? # TODO: set - maybe use check_numerical
        end

        @testset "model_2" begin
            logprior_2 = gen_logprior(spl_2.state.vi, model_2, spl_2)
            # @test logprior_2([???]) == ??? # TODO: set - maybe use check_numerical
            # @test logprior_2([???]) == ??? # TODO: set - maybe use check_numerical
            # @test logprior_2([???]) == ??? # TODO: set - maybe use check_numerical
        end
    end

    @turing_testset "gen_log_unnorm_tempered" begin
        beta = 0.5
        @testset "model_1" begin
            logjoint_1 = gen_logjoint(spl_1.state.vi, model_1, spl_1)
            logprior_1 = gen_logprior(spl_1.state.vi, model_1, spl_1)
            log_unnorm_tempered_1 = gen_log_unnorm_tempered(logprior_1, logjoint_1, beta)
            # @test log_unnorm_tempered_1([???]) == ??? # TODO: set - maybe use check_numerical
            # @test log_unnorm_tempered_1([???]) == ??? # TODO: set - maybe use check_numerical
            # @test log_unnorm_tempered_1([???]) == ??? # TODO: set - maybe use check_numerical
        end

        @testset "model_2" begin
            logjoint_2 = gen_logjoint(spl_2.state.vi, model_2, spl_2)
            logprior_2 = gen_logprior(spl_2.state.vi, model_2, spl_2)
            log_unnorm_tempered_2 = gen_log_unnorm_tempered(logprior_2, logjoint_2, beta)
            # @test log_unnorm_tempered_2([???]) == ??? # TODO: set - maybe use check_numerical
            # @test log_unnorm_tempered_2([???]) == ??? # TODO: set - maybe use check_numerical
            # @test log_unnorm_tempered_2([???]) == ??? # TODO: set - maybe use check_numerical
        end
    end

    @testset "plots for gen_logjoint, gen_logprior, gen_log_unnorm_tempered" begin
        interval = 0:0.1:10

        # test gen_logjoint for model 2: plot
        logjoint = gen_logjoint(spl_2.state.vi, model_2, spl_2)
        logjoint_values = logjoint.([[x] for x in interval])
        density_plots = plot(interval, logjoint_values)

        # test gen_logprior for model 2: plot
        logprior = gen_logprior(spl_2.state.vi, model_2, spl_2)
        logprior_values = logprior.([[x] for x in interval])
        plot!(density_plots, interval, logprior_values)

        # test gen_log_unnorm_tempered for model 2: plot
        for beta in 0.1:0.1:0.9
            log_unnorm_tempered = gen_log_unnorm_tempered(logprior, logjoint, beta)
            log_unnorm_tempered_values = log_unnorm_tempered.([[x] for x in interval])
            plot!(density_plots, interval, log_unnorm_tempered_values)
        end

        display(density_plots)
    end

    # 3. tests related to sample_init!

    @testset "sample_init!" begin
        @testset "model_1" begin
            @test length(spl_1.state.densitymodels) == 0
            sample_init!(MersenneTwister(1234), model_1, spl_1, 1)
            @test length(spl_1.state.densitymodels) > 0
        end

        @testset "model_2" begin
            @test length(spl_2.state.densitymodels) == 0
            sample_init!(MersenneTwister(1234), model_2, spl_2, 1)
            @test length(spl_2.state.densitymodels) > 0
        end
    end

    # 4. tests related to step!

    @testset "prior_step" begin
        @testset "model_1" begin
        end 

        @testset "model_2" begin
        end
    end
    
    @testset "plots for prior_step" begin
        list_samples = []
        for i in 1:50
            push!(list_samples, first(prior_step(spl_2, model_2)))
        end
        prior_step_hist = histogram(list_samples)
        display(prior_step_hist)
    end
    
    @testset "intermediate_step" begin
        @testset "model_1" begin
        end 

        @testset "model_2" begin
        end
    end

    @testset "step!" begin
        @testset "model_1" begin
        end 

        @testset "model_2" begin
        end
    end

    # 5. tests related to sample_end! 

    @testset "sample_end!" begin
        @testset "model_1" begin
        end 

        @testset "model_2" begin
        end
    end

    # 6. test general performance
    @testset "general performance" begin
        @testset "model_1" begin
            # what is the logevidence
        end 

        @testset "model_2" begin
        end
    end 
end