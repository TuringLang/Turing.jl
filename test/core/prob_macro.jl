using Turing, Distributions, Test, Random

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

Random.seed!(129)

@turing_testset "prob_macro" begin
    @testset "scalar" begin
        @model demo(x) = begin
            m ~ Normal()
            x ~ Normal(m, 1)
        end

        mval = 3
        xval = 2
        iters = 1000

        logprior = logpdf(Normal(), mval)
        loglike = logpdf(Normal(mval, 1), xval)
        logjoint = logprior + loglike

        @test logprob"m = mval | model = demo" == logprior
        @test logprob"x = xval | m = mval, model = demo" == loglike
        @test logprob"x = xval, m = mval | model = demo" == logjoint

        varinfo = Turing.VarInfo(demo(xval))
        @test logprob"m = mval | model = demo, varinfo = varinfo" == logprior
        @test logprob"x = xval | m = mval, model = demo, varinfo = varinfo" == loglike
        varinfo = Turing.VarInfo(demo(missing))
        @test logprob"x = xval, m = mval | model = demo, varinfo = varinfo" == logjoint

        chain = sample(demo(xval), IS(), iters)
        lps = logpdf.(Normal.(vec(chain["m"].value), 1), xval)
        @test logprob"x = xval | chain = chain, model = demo" == lps
    end

    @testset "vector" begin
        n = 5
        @model demo(x, n = n, ::Type{T} = Float64) where {T} = begin
            m = Vector{T}(undef, n)
            @. m ~ Normal()
            @. x ~ Normal.(m, 1)
        end
        mval = rand(n)
        xval = rand(n)
        iters = 1000

        logprior = sum(logpdf.(Normal(), mval))
        like(m, x) = sum(logpdf.(Normal.(m, 1), x))
        loglike = like(mval, xval)
        logjoint = logprior + loglike

        @test logprob"m = mval | model = demo" == logprior
        @test logprob"x = xval | m = mval, model = demo" == loglike
        @test logprob"x = xval, m = mval | model = demo" == logjoint

        varinfo = Turing.VarInfo(demo(xval))
        @test logprob"m = mval | model = demo, varinfo = varinfo" == logprior
        @test logprob"x = xval | m = mval, model = demo, varinfo = varinfo" == loglike
        # Currently, we cannot easily pre-allocate `VarInfo` for vector data

        chain = sample(demo(xval), HMC(0.5, 1), iters)
        lps = like.([[chain["m[$i]"].value[j] for i in 1:n] for j in 1:iters], Ref(xval))
        @test logprob"x = xval | chain = chain, model = demo" == lps
    end
end
