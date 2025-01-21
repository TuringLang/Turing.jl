module AdTests

using ..Models: gdemo_default
using Distributions: logpdf
using DynamicPPL: DynamicPPL, getlogp, getindex_internal
using ForwardDiff
using LinearAlgebra
using LogDensityProblems: LogDensityProblems
using LogDensityProblemsAD: LogDensityProblemsAD
using ReverseDiff
using Test: @test, @testset
using Turing
using Turing: SampleFromPrior
using Zygote

function test_model_ad(model, f, syms::Vector{Symbol})
    # Set up VI.
    vi = Turing.VarInfo(model)

    # Collect symbols.
    vnms = Vector(undef, length(syms))
    vnvals = Vector{Float64}()
    for i in 1:length(syms)
        s = syms[i]
        vnms[i] = getfield(vi.metadata, s).vns[1]

        vals = getindex_internal(vi, vnms[i])
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
            Turing.AutoForwardDiff(; chunksize=chunksize, tag=standardtag),
            DynamicPPL.LogDensityFunction(
                vi, model, SampleFromPrior(), DynamicPPL.DefaultContext()
            ),
        )
        l, ∇E = LogDensityProblems.logdensity_and_gradient(ℓ, z)

        # Compare result
        @test l ≈ logp
        @test sort(∇E) ≈ grad_FWAD atol = 1e-9
    end
end

@testset "ad.jl" begin
    @testset "adr" begin
        ad_test_f = gdemo_default
        vi = Turing.VarInfo(ad_test_f)
        ad_test_f(vi, SampleFromPrior())
        svn = vi.metadata.s.vns[1]
        mvn = vi.metadata.m.vns[1]
        _s = getindex_internal(vi, svn)[1]
        _m = getindex_internal(vi, mvn)[1]

        dist_s = InverseGamma(2, 3)

        # Hand-written logp
        function logp(x::Vector)
            s = x[2]
            # s = invlink(dist_s, s)
            m = x[1]
            lik_dist = Normal(m, sqrt(s))
            lp = logpdf(dist_s, s) + logpdf(Normal(0, sqrt(s)), m)
            lp += logpdf(lik_dist, 1.5) + logpdf(lik_dist, 2.0)
            return lp
        end

        # Call ForwardDiff's AD
        g = x -> ForwardDiff.gradient(logp, x)
        # _s = link(dist_s, _s)
        _x = [_m, _s]
        grad_FWAD = sort(g(_x))

        ℓ = DynamicPPL.LogDensityFunction(
            vi, ad_test_f, SampleFromPrior(), DynamicPPL.DefaultContext()
        )
        x = map(x -> Float64(x), vi[SampleFromPrior()])

        zygoteℓ = LogDensityProblemsAD.ADgradient(Turing.AutoZygote(), ℓ)
        if isdefined(Base, :get_extension)
            @test zygoteℓ isa
                Base.get_extension(
                LogDensityProblemsAD, :LogDensityProblemsADZygoteExt
            ).ZygoteGradientLogDensity
        else
            @test zygoteℓ isa
                LogDensityProblemsAD.LogDensityProblemsADZygoteExt.ZygoteGradientLogDensity
        end
        @test zygoteℓ.ℓ === ℓ
        ∇E2 = LogDensityProblems.logdensity_and_gradient(zygoteℓ, x)[2]
        @test sort(∇E2) ≈ grad_FWAD atol = 1e-9
    end

    @testset "general AD tests" begin
        # Tests gdemo gradient.
        function logp1(x::Vector)
            dist_s = InverseGamma(2, 3)
            s = x[2]
            m = x[1]
            lik_dist = Normal(m, sqrt(s))
            lp =
                Turing.logpdf_with_trans(dist_s, s, false) +
                Turing.logpdf_with_trans(Normal(0, sqrt(s)), m, false)
            lp += logpdf(lik_dist, 1.5) + logpdf(lik_dist, 2.0)
            return lp
        end

        test_model_ad(gdemo_default, logp1, [:m, :s])

        # Test Wishart AD.
        @model function wishart_ad()
            v ~ Wishart(7, [1 0.5; 0.5 1])
            return v
        end

        # Hand-written logp
        function logp3(x)
            dist_v = Wishart(7, [1 0.5; 0.5 1])
            v = [x[1] x[3]; x[2] x[4]]
            lp = Turing.logpdf_with_trans(dist_v, v, false)
            return lp
        end

        test_model_ad(wishart_ad(), logp3, [:v])
    end
    @testset "Simplex Zygote and ReverseDiff (with and without caching) AD" begin
        @model function dir()
            return theta ~ Dirichlet(1 ./ fill(4, 4))
        end
        sample(dir(), HMC(0.01, 1; adtype=AutoZygote()), 1000)
        sample(dir(), HMC(0.01, 1; adtype=AutoReverseDiff(; compile=false)), 1000)
        sample(dir(), HMC(0.01, 1; adtype=AutoReverseDiff(; compile=true)), 1000)
    end
    @testset "PDMatDistribution AD" begin
        @model function wishart()
            return theta ~ Wishart(4, Matrix{Float64}(I, 4, 4))
        end

        sample(wishart(), HMC(0.01, 1; adtype=AutoReverseDiff(; compile=false)), 1000)
        sample(wishart(), HMC(0.01, 1; adtype=AutoZygote()), 1000)

        @model function invwishart()
            return theta ~ InverseWishart(4, Matrix{Float64}(I, 4, 4))
        end

        sample(invwishart(), HMC(0.01, 1; adtype=AutoReverseDiff(; compile=false)), 1000)
        sample(invwishart(), HMC(0.01, 1; adtype=AutoZygote()), 1000)
    end
    @testset "Hessian test" begin
        @model function tst(x, ::Type{TV}=Vector{Float64}) where {TV}
            params = TV(undef, 2)
            @. params ~ Normal(0, 1)

            return x ~ MvNormal(params, I)
        end

        function make_logjoint(model::DynamicPPL.Model, ctx::DynamicPPL.AbstractContext)
            # setup
            varinfo_init = Turing.VarInfo(model)
            spl = DynamicPPL.SampleFromPrior()
            varinfo_init = DynamicPPL.link!!(varinfo_init, spl, model)

            function logπ(z; unlinked=false)
                varinfo = DynamicPPL.unflatten(varinfo_init, spl, z)

                # TODO(torfjelde): Pretty sure this is a mistake.
                # Why are we not linking `varinfo` rather than `varinfo_init`?
                if unlinked
                    varinfo_init = DynamicPPL.invlink!!(varinfo_init, spl, model)
                end
                varinfo = last(
                    DynamicPPL.evaluate!!(
                        model, varinfo, DynamicPPL.SamplingContext(spl, ctx)
                    ),
                )
                if unlinked
                    varinfo_init = DynamicPPL.link!!(varinfo_init, spl, model)
                end

                return -DynamicPPL.getlogp(varinfo)
            end

            return logπ
        end

        data = [0.5, -0.5]
        model = tst(data)

        likelihood = make_logjoint(model, DynamicPPL.LikelihoodContext())
        target(x) = likelihood(x; unlinked=true)

        H_f = ForwardDiff.hessian(target, zeros(2))
        H_r = ReverseDiff.hessian(target, zeros(2))
        @test H_f == [1.0 0.0; 0.0 1.0]
        @test H_f == H_r
    end

    @testset "memoization: issue #1393" begin
        @model function demo(data)
            sigma ~ Uniform(0.0, 20.0)
            return data ~ Normal(0, sigma)
        end

        N = 1000
        for i in 1:5
            d = Normal(0.0, i)
            data = rand(d, N)
            chn = sample(
                demo(data), NUTS(0.65; adtype=AutoReverseDiff(; compile=true)), 1000
            )
            @test mean(Array(chn[:sigma])) ≈ std(data) atol = 0.5
        end
    end

    @testset "ReverseDiff compiled without linking" begin
        f = DynamicPPL.LogDensityFunction(gdemo_default)
        θ = DynamicPPL.getparams(f)

        f_rd = LogDensityProblemsAD.ADgradient(Turing.AutoReverseDiff(; compile=false), f)
        f_rd_compiled = LogDensityProblemsAD.ADgradient(
            Turing.AutoReverseDiff(; compile=true), f
        )

        ℓ, ℓ_grad = LogDensityProblems.logdensity_and_gradient(f_rd, θ)
        ℓ_compiled, ℓ_grad_compiled = LogDensityProblems.logdensity_and_gradient(
            f_rd_compiled, θ
        )

        @test ℓ == ℓ_compiled
        @test ℓ_grad == ℓ_grad_compiled
    end
end

end
