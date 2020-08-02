# TODO: lots of imports will have been done in runtests.jl hence irrelevant
using Zygote, ReverseDiff, Memoization, Turing; turnprogress(false)
using Turing: AIS, Sampler
using Pkg
using Random
using Test
using DynamicPPL: getlogp, setlogp!, SampleFromPrior, PriorContext, VarInfo
using Plots
using AdvancedMH

# part of the header for all tests in inference
dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@turing_testset "ais.jl" begin

    # 1. declare models and associated samplers

    # model_1: normal distributions, but multidimensional
    @model model_1_macro(x, y) begin
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
    proposal_kernels_1 = [AdvancedMH.RWMH([Normal(), Normal()], Normal()) for i in 1:9]
    alg_1 = AIS(proposal_kernels_1, schedule_1)
    spl_1 = Sampler(alg_1, model_1)

    # model_2: non-normal distributions, but unidimensional
    @model model_2_macro(x) begin
        # latent
        inv_theta ~ Gamma(2,3)
        theta = 1/inv_theta
        # observed
        x ~ Weibull(1,theta) 
    end
    model_2 = model_2_macro(5.)

    # declare algorithm and sampler for model_2
    schedule_2 = 0.1:0.1:0.9
    proposal_kernels_2 = [AdvancedMH.RWMH([Normal(), Normal()], Normal()) for i in 1:9]
    alg_2 = AIS(proposal_kernels_2, schedule_2)
    spl_2 = Sampler(alg_2, model_2)

    # 2. test lower level functions: gen_logjoint, gen_logprior, gen_log_unnorm_tempered

    @turing_testset "gen_logjoint" begin
        println("model_1:")
        println("model_2:")
    end

    @turing_testset "gen_logprior" begin
        println("model_1:")
        println("model_2:")
    end

    @turing_testset "gen_log_unnorm_tempered" begin
        println("model_1:")
        println("model_2:")
    end

    @testset "plots for gen_logjoint, gen_logprior, gen_log_unnorm_tempered" begin
        interval = 0:0.1:10

        # test gen_logjoint for model 2: plot
        logjoint = gen_logjoint(spl_2.state.vi, model_2, spl_2)
        logjoint_values = logjoint.(list_args)
        density_plots = plot(interval, logjoint_values)

        # test gen_logprior for model 2: plot
        logprior = gen_logprior(spl_2.state.vi, model_2, spl_2)
        logprior_values = logprior.(list_args)
        plot!(density_plots, interval, logprior_values)

        # test gen_log_unnorm_tempered for model 2: plot
        for beta in 0.1:0.1:0.9
            log_unnorm_tempered = gen_log_unnorm_tempered(logprior, logjoint, beta)
            log_unnorm_tempered_values = log_unnorm_tempered.(list_args)
            plot!(density_plots, interval, log_unnorm_tempered_values)
        end

        display(density_plots)
    end

    # 3. test mid level functions: prior_step, intermediate_step

    @testset "prior_step" begin
        println("model_1:")
        println("model_2:")
    end
    
    @testset "plots for prior_step" begin
        list_samples = []
        for i in 1:50
            append!(list_samples, prior_step(model_2)[1]) # TODO: prior_step(model) currently returns an array, probably shouldn't
        end
        prior_step_hist = histogram(list_samples)

        display(prior_step_hist)
    end
    
    @testset "intermediate_step" begin
        println("model_1:") 
        println("model_2:")
    end

    # 4. test high level functions: implementations of AbstractMCMC, ie sample_init!, step!, sample_end! 

    @testset "sample_init!" begin
        println("model_1:")
        println("model_2:")
    end

    @testset "step!" begin
        println("model_1:")
        println("model_2:")
    end

    @testset "sample_end!" begin
        println("model_1:")
        println("model_2:")
    end
end