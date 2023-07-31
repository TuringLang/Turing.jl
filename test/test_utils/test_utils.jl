#
# Load dependencies
#

using AbstractMCMC
using AdvancedMH
using Clustering
using Distributions
using Distributions.FillArrays
using DistributionsAD
using FiniteDifferences
using ForwardDiff
using MCMCChains
using NamedArrays
using Optim
using Optimization
using OptimizationOptimJL
using PDMats
using ReverseDiff
using SpecialFunctions
using StatsBase
using StatsFuns
using Tracker
using Turing
using Turing.Inference
using Turing.RandomMeasures
using Zygote

using LinearAlgebra
using Pkg
using Random
using Test
using StableRNGs

using AdvancedPS: ResampleWithESSThreshold, resample_systematic, resample_multinomial
using AdvancedVI: TruncatedADAGrad, DecayedADAGrad, apply!
using Distributions: Binomial, logpdf
using DynamicPPL: getval, getlogp
using ForwardDiff: Dual
using MCMCChains: Chains
using StatsFuns: binomlogpdf, logistic, logsumexp
using TimerOutputs: TimerOutputs, @timeit
using Turing: BinomialLogit, ForwardDiffAD, Sampler, SampleFromPrior, NUTS, TrackerAD,
                Variational, ZygoteAD, getspace
using Turing.Variational: AdvancedVI
using Turing.Essential: TuringDenseMvNormal, TuringDiagMvNormal
import LogDensityProblems
import LogDensityProblemsAD


#
# Staging for tests
#

function get_stage()
    # Appveyor uses "True" for non-Ubuntu images.
    if get(ENV, "APPVEYOR", "") == "True" || get(ENV, "APPVEYOR", "") == "true"
        return "nonnumeric"
    end

    # Handle Travis and Github Actions specially.
    if get(ENV, "TRAVIS", "") == "true" || get(ENV, "GITHUB_ACTIONS", "") == "true"
        if "STAGE" in keys(ENV)
            return ENV["STAGE"]
        else
            return "all"
        end
    end

    return "all"
end

function do_test(stage_str)
    stg = get_stage()

    # If the tests are being run by Appveyor, don't run
    # any numerical tests.
    if stg == "nonnumeric"
        if stage_str == "numerical"
            return false
        else
            return true
        end
    end

    # Otherwise run the regular testing procedure.
    if stg == "all" || stg == stage_str
        return true
    end

    return false
end

macro stage_testset(stage_string::String, args...)
    if do_test(stage_string)
        return esc(:(@testset($(args...))))
    end
end

macro numerical_testset(args...)
    esc(:(@stage_testset "numerical" $(args...)))
end

macro turing_testset(args...)
    esc(:(@stage_testset "test" $(args...)))
end


#
# Autodiff 
#

"""
    test_reverse_mode_ad(forward, f, ȳ, x...; rtol=1e-6, atol=1e-6)

Check that the reverse-mode sensitivities produced by an AD library are correct for `f`
at `x...`, given sensitivity `ȳ` w.r.t. `y = f(x...)` up to `rtol` and `atol`.
"""
function test_reverse_mode_ad( f, ȳ, x...; rtol=1e-6, atol=1e-6)
    # Perform a regular forwards-pass.
    y = f(x...)

    # Use Tracker to compute reverse-mode sensitivities.
    y_tracker, back_tracker = Tracker.forward(f, x...)
    x̄s_tracker = back_tracker(ȳ)

    # Use Zygote to compute reverse-mode sensitivities.
    y_zygote, back_zygote = Zygote.pullback(f, x...)
    x̄s_zygote = back_zygote(ȳ)

    test_rd = length(x) == 1 && y isa Number
    if test_rd
        # Use ReverseDiff to compute reverse-mode sensitivities.
        if x[1] isa Array
            x̄s_rd = similar(x[1])
            tp = ReverseDiff.GradientTape(x -> f(x), x[1])
            ReverseDiff.gradient!(x̄s_rd, tp, x[1])
            x̄s_rd .*= ȳ
            y_rd = ReverseDiff.value(tp.output)
            @assert y_rd isa Number
        else
            x̄s_rd = [x[1]]
            tp = ReverseDiff.GradientTape(x -> f(x[1]), [x[1]])
            ReverseDiff.gradient!(x̄s_rd, tp, [x[1]])
            y_rd = ReverseDiff.value(tp.output)[1]
            x̄s_rd = x̄s_rd[1] * ȳ
            @assert y_rd isa Number
        end
    end

    # Use finite differencing to compute reverse-mode sensitivities.
    x̄s_fdm = FDM.j′vp(central_fdm(5, 1), f, ȳ, x...)

    # Check that Tracker forwards-pass produces the correct answer.
    @test isapprox(y, Tracker.data(y_tracker), atol=atol, rtol=rtol)

    # Check that Zygpte forwards-pass produces the correct answer.
    @test isapprox(y, y_zygote, atol=atol, rtol=rtol)

    if test_rd
        # Check that ReverseDiff forwards-pass produces the correct answer.
        @test isapprox(y, y_rd, atol=atol, rtol=rtol)
    end

    # Check that Tracker reverse-mode sensitivities are correct.
    @test all(zip(x̄s_tracker, x̄s_fdm)) do (x̄_tracker, x̄_fdm)
        isapprox(Tracker.data(x̄_tracker), x̄_fdm; atol=atol, rtol=rtol)
    end

    # Check that Zygote reverse-mode sensitivities are correct.
    @test all(zip(x̄s_zygote, x̄s_fdm)) do (x̄_zygote, x̄_fdm)
        isapprox(x̄_zygote, x̄_fdm; atol=atol, rtol=rtol)
    end

    if test_rd
        # Check that ReverseDiff reverse-mode sensitivities are correct.
        @test isapprox(x̄s_rd, x̄s_zygote[1]; atol=atol, rtol=rtol)
    end
end

function test_model_ad(model, f, syms::Vector{Symbol})
    # Set up VI.
    vi = Turing.VarInfo(model)

    # Collect symbols.
    vnms = Vector(undef, length(syms))
    vnvals = Vector{Float64}()
    for i in 1:length(syms)
        s = syms[i]
        vnms[i] = getfield(vi.metadata, s).vns[1]

        vals = getval(vi, vnms[i])
        for i in eachindex(vals)
            push!(vnvals, vals[i])
        end
    end

    # Compute primal.
    x = vec(vnvals)
    logp = f(x)

    # Call ForwardDiff's AD directly.
    grad_FWAD = sort(ForwardDiff.gradient(f, x))

    # Compare with `logdensity_and_gradient`.
    z = vi[SampleFromPrior()]
    for chunksize in (0, 1, 10), standardtag in (true, false, 0, 3)
        ℓ = LogDensityProblemsAD.ADgradient(
            ForwardDiffAD{chunksize, standardtag}(),
            Turing.LogDensityFunction(vi, model, SampleFromPrior(), DynamicPPL.DefaultContext()),
        )
        l, ∇E = LogDensityProblems.logdensity_and_gradient(ℓ, z)

        # Compare result
        @test l ≈ logp
        @test sort(∇E) ≈ grad_FWAD atol=1e-9
    end
end

#
# Numerical related functions
#

function check_dist_numerical(dist, chn; mean_tol = 0.1, var_atol = 1.0, var_tol = 0.5)
    @testset "numerical" begin
        # Extract values.
        chn_xs = Array(chn[1:2:end, namesingroup(chn, :x), :])

        # Check means.
        dist_mean = mean(dist)
        mean_shape = size(dist_mean)
        if !all(isnan, dist_mean) && !all(isinf, dist_mean)
            chn_mean = vec(mean(chn_xs, dims=1))
            chn_mean = length(chn_mean) == 1 ?
                chn_mean[1] :
                reshape(chn_mean, mean_shape)
            atol_m = length(chn_mean) > 1 ?
                mean_tol * length(chn_mean) :
                max(mean_tol, mean_tol * chn_mean)
            @test chn_mean ≈ dist_mean atol=atol_m
        end

        # Check variances.
        # var() for Distributions.MatrixDistribution is not defined
        if !(dist isa Distributions.MatrixDistribution)
            # Variance
            dist_var = var(dist)
            var_shape = size(dist_var)
            if !all(isnan, dist_var) && !all(isinf, dist_var)
                chn_var = vec(var(chn_xs, dims=1))
                chn_var = length(chn_var) == 1 ?
                    chn_var[1] :
                    reshape(chn_var, var_shape)
                atol_v = length(chn_mean) > 1 ?
                    mean_tol * length(chn_mean) :
                    max(mean_tol, mean_tol * chn_mean)
                @test chn_mean ≈ dist_mean atol=atol_v
            end
        end
    end
end

# Helper function for numerical tests
function check_numerical(chain,
                        symbols::Vector,
                        exact_vals::Vector;
                        atol=0.2,
                        rtol=0.0)
    for (sym, val) in zip(symbols, exact_vals)
        E = val isa Real ?
            mean(chain[sym]) :
            vec(mean(chain[sym], dims=1))
        @info (symbol=sym, exact=val, evaluated=E)
        @test E ≈ val atol=atol rtol=rtol
    end
end

#
# Various testing related functions
#

GKernel(var) = (x) -> Normal(x, sqrt.(var))

function randr(vi::Turing.VarInfo,
                vn::Turing.VarName,
                dist::Distribution,
                spl::Turing.Sampler,
                count::Bool = false)
    if ~haskey(vi, vn)
        r = rand(dist)
        Turing.push!(vi, vn, r, dist, spl)
        return r
    elseif is_flagged(vi, vn, "del")
        unset_flag!(vi, vn, "del")
        r = rand(dist)
        Turing.RandomVariables.setval!(vi, Turing.vectorize(dist, r), vn)
        return r
    else
        if count Turing.checkindex(vn, vi, spl) end
        Turing.updategid!(vi, vn, spl)
        return vi[vn]
    end
end

function insdelim(c, deli=",")
return reduce((e, res) -> append!(e, [res, deli]), c; init = [])[1:end-1]
end

#
# Import utility functions and reused models.
#
include("models.jl")
