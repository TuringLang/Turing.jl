@testset "advi.jl" begin
    @turing_testset "advi constructor" begin
        Random.seed!(0)
        N = 500

        s1 = ADVI()
        q = vi(gdemo_default, s1)
        c1 = rand(q, N)
    end
    @numerical_testset "advi inference" begin
        @testset for opt in [TruncatedADAGrad(), DecayedADAGrad()]
            Random.seed!(1)
            N = 500

            alg = ADVI(10, 5000)
            q = vi(gdemo_default, alg; optimizer = opt)
            samples = transpose(rand(q, N))
            chn = Chains(reshape(samples, size(samples)..., 1), ["s", "m"])

            # TODO: uhmm, seems like a large `eps` here...
            check_gdemo(chn, atol = 0.5)
        end
    end

    @turing_testset "advi different interfaces" begin
        Random.seed!(1234)

        target = MvNormal(zeros(2), I)
        logπ(z) = logpdf(target, z)
        advi = ADVI(10, 1000)

        # Using a function z ↦ q(⋅∣z)
        getq(θ) = TuringDiagMvNormal(θ[1:2], exp.(θ[3:4]))
        q = vi(logπ, advi, getq, randn(4))

        xs = rand(target, 10)
        @test mean(abs2, logpdf(q, xs) - logpdf(target, xs)) ≤ 0.07

        # OR: implement `update` and pass a `Distribution`
        function AdvancedVI.update(d::TuringDiagMvNormal, θ::AbstractArray{<:Real})
            return TuringDiagMvNormal(θ[1:length(q)], exp.(θ[length(q) + 1:end]))
        end

        q0 = TuringDiagMvNormal(zeros(2), ones(2))
        q = vi(logπ, advi, q0, randn(4))

        xs = rand(target, 10)
        @test mean(abs2, logpdf(q, xs) - logpdf(target, xs)) ≤ 0.05
    end

    # regression test for:
    # https://github.com/TuringLang/Turing.jl/issues/2065
    @turing_testset "simplex bijector" begin
        @model function dirichlet()
            x ~ Dirichlet([1.0,1.0])
            return x
        end
        
        m = dirichlet()
        b = bijector(m)
        x0 = m()
        z0 = b(x0)
        @test size(z0) == (1,)
        x0_inv = inverse(b)(z0)
        @test size(x0_inv) == size(x0)
        @test all(x0 .≈ x0_inv)

        # And regression for https://github.com/TuringLang/Turing.jl/issues/2160.
        q = vi(m, ADVI(10, 1000))
        x = rand(q, 1000)
        @test mean(eachcol(x)) ≈ [0.5, 0.5] atol=0.1
    end
end
