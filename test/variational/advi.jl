@testset "advi.jl" begin
    @turing_testset "advi constructor" begin
        Random.seed!(0)
        s1 = ADVI()
        q  = vi(gdemo_default, s1, 1000)
        c1 = rand(q, 500)
    end

    @numerical_testset "advi inference" begin
        opt = Adam()
        Random.seed!(1)
        N = 500

        alg = ADVI(10)
        q = vi(gdemo_default, alg, 5000; optimizer = opt)
        samples = transpose(rand(q, N))
        chn = Chains(reshape(samples, size(samples)..., 1), ["s", "m"])

        # TODO: uhmm, seems like a large `eps` here...
        check_gdemo(chn, atol = 0.5)
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
end
