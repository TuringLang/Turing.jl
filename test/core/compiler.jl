using Turing, Random, MacroTools, Distributions, Test
using Turing.Core: split_var_str

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

Random.seed!(129)

priors = 0 # See "new grammar" test.

@turing_testset "compiler.jl" begin
    @testset "assume" begin
        @model test_assume() = begin
            x ~ Bernoulli(1)
            y ~ Bernoulli(x / 2)
            x, y
        end

        smc = SMC()
        pg = PG(10)

        res1 = sample(test_assume(), smc, 1000)
        res2 = sample(test_assume(), pg, 1000)

        check_numerical(res1, [:y], [0.5], atol=0.1)
        check_numerical(res2, [:y], [0.5], atol=0.1)

        # Check that all xs are 1.
        @test all(isone, res1[:x].value)
        @test all(isone, res2[:x].value)
    end
    @testset "beta binomial" begin
        prior = Beta(2,2)
        obs = [0,1,0,1,1,1,1,1,1,1]
        exact = Beta(prior.α + sum(obs), prior.β + length(obs) - sum(obs))
        meanp = exact.α / (exact.α + exact.β)

        @model testbb(obs) = begin
            p ~ Beta(2,2)
            x ~ Bernoulli(p)
            for i = 1:length(obs)
                obs[i] ~ Bernoulli(p)
            end
            p, x
        end

        smc = SMC()
        pg = PG(10)
        gibbs = Gibbs(HMC(0.2, 3, :p), PG(10, :x))

        chn_s = sample(testbb(obs), smc, 1000)
        chn_p = sample(testbb(obs), pg, 2000)
        chn_g = sample(testbb(obs), gibbs, 1500)

        check_numerical(chn_s, [:p], [meanp], atol=0.05)
        check_numerical(chn_p, [:x], [meanp], atol=0.1)
        check_numerical(chn_g, [:x], [meanp], atol=0.1)
    end
    @testset "forbid missing inputs" begin

    end
    @testset "forbid global" begin
        xs = [1.5 2.0]
        # xx = 1

        @model fggibbstest(xs) = begin
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))
            # xx ~ Normal(m, sqrt(s)) # this is illegal

            for i = 1:length(xs)
                xs[i] ~ Normal(m, sqrt(s))
                # for xx in xs
                # xx ~ Normal(m, sqrt(s))
            end
            s, m
        end

        gibbs = Gibbs(PG(10, :s), HMC(0.4, 8, :m))
        chain = sample(fggibbstest(xs), gibbs, 2);
    end
    @testset "model macro" begin
        @model testmodel_comp(x, y) = begin
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))

            x ~ Normal(m, sqrt(s))
            y ~ Normal(m, sqrt(s))

            return x, y
        end
        testmodel_comp(1.0, 1.2)

        # check if drawing from the prior works
        @model testmodel0(x = missing) = begin
            x ~ Normal()
            return x
        end
        f0_mm = testmodel0()
        @test mean(f0_mm() for _ in 1:1000) ≈ 0. atol=0.1

        # Test #544
        @model testmodel0(x = missing) = begin
            if x === missing
                x = Vector{Float64}(undef, 2)
            end
            x[1] ~ Normal()
            x[2] ~ Normal()
            return x
        end
        f0_mm = testmodel0()
        @test all(x -> isapprox(x, 0; atol = 0.1), mean(f0_mm() for _ in 1:1000))

        @model testmodel01(x = missing) = begin
            x ~ Bernoulli(0.5)
            return x
        end
        f01_mm = testmodel01()
        @test mean(f01_mm() for _ in 1:1000) ≈ 0.5 atol=0.1

        # test if we get the correct return values
        @model testmodel1(x1, x2) = begin
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))

            x1 ~ Normal(m, sqrt(s))
            x2 ~ Normal(m, sqrt(s))

            return x1, x2
        end
        f1_mm = testmodel1(1., 10.)
        @test f1_mm() == (1, 10)
        f1_mm = testmodel1(x1=1., x2=10.)
        @test f1_mm() == (1, 10)

        @info "Testing the compiler's ability to catch bad models..."

        # Test for assertions in observe statements.
        @model brokentestmodel_observe1(x1, x2) = begin
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))

            x1 ~ Normal(m, sqrt(s))
            x2 ~ x1 + 2

            return x1, x2
        end

        btest = brokentestmodel_observe1(1., 2.)
        @test_throws ArgumentError btest()

        @model brokentestmodel_observe2(x) = begin
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))

            x = Vector{Float64}(undef, 2)
            x ~ [Normal(m, sqrt(s)), 2.0]

            return x
        end

        btest = brokentestmodel_observe2([1., 2.])
        @test_throws ArgumentError btest()

        # Test for assertions in assume statements.
        @model brokentestmodel_assume1() = begin
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))

            x1 ~ Normal(m, sqrt(s))
            x2 ~ x1 + 2

            return x1, x2
        end

        btest = brokentestmodel_assume1()
        @test_throws ArgumentError btest()

        @model brokentestmodel_assume2() = begin
            s ~ InverseGamma(2,3)
            m ~ Normal(0,sqrt(s))

            x = Vector{Float64}(undef, 2)
            x ~ [Normal(m, sqrt(s)), 2.0]

            return x
        end

        btest = brokentestmodel_assume2()
        @test_throws ArgumentError btest()

        # Test missing input arguments
        @model testmodel(x) = begin
            x ~ Bernoulli(0.5)
            return x
        end
        @test_throws UndefKeywordError testmodel()

        # Test missing initialization for vector observation turned parameter
        @model testmodel(x) = begin
            x[1] ~ Bernoulli(0.5)
            return x
        end
        @test_throws MethodError testmodel(missing)()

        # Test @varinfo() and @logpdf()
        @model testmodel(x) = begin
            x[1] ~ Bernoulli(0.5)
            global _varinfo = @varinfo()
            global lp = @logpdf()
            return x
        end
        model = testmodel([1.0])
        varinfo = Turing.VarInfo(model)
        model(varinfo)
        @test varinfo.logp == lp
        @test varinfo === _varinfo
    end
    @testset "new grammar" begin
        x = Float64[1 2]

        @model gauss(x) = begin
            priors = TArray{Float64}(2)
            priors[1] ~ InverseGamma(2,3)         # s
            priors[2] ~ Normal(0, sqrt(priors[1])) # m
            for i in 1:length(x)
                x[i] ~ Normal(priors[2], sqrt(priors[1]))
            end
            priors
        end

        chain = sample(gauss(x), PG(10), 10)
        chain = sample(gauss(x), SMC(), 10)

        @model gauss2(x, ::Type{TV}=Vector{Float64}) where {TV} = begin
            priors = TV(undef, 2)
            priors[1] ~ InverseGamma(2,3)         # s
            priors[2] ~ Normal(0, sqrt(priors[1])) # m
            for i in 1:length(x)
                x[i] ~ Normal(priors[2], sqrt(priors[1]))
            end
            priors
        end

        chain = sample(gauss2(x), PG(10), 10)
        chain = sample(gauss2(x=x, TV=Vector{Float64}), PG(10), 10)
        chain = sample(gauss2(x), SMC(), 10)
        chain = sample(gauss2(x=x, TV=Vector{Float64}), SMC(), 10)
    end
    @testset "new interface" begin
        obs = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1]

        @model newinterface(obs) = begin
          p ~ Beta(2,2)
          for i = 1:length(obs)
            obs[i] ~ Bernoulli(p)
          end
          p
        end

        chain = sample(
            newinterface(obs),
            HMC{Turing.ForwardDiffAD{2}}(0.75, 3, :p, :x),
            100)
    end
    @testset "no return" begin
        @model noreturn(x) = begin
            s ~ InverseGamma(2,3)
            m ~ Normal(0, sqrt(s))
            for i in 1:length(x)
                x[i] ~ Normal(m, sqrt(s))
            end
        end

        chain = sample(noreturn([1.5 2.0]), HMC(0.15, 6), 1000)
        check_numerical(chain, [:s, :m], [49/24, 7/6])
    end
    @testset "observe" begin
        @model test() = begin
          z ~ Normal(0,1)
          x ~ Bernoulli(1)
          1 ~ Bernoulli(x / 2)
          0 ~ Bernoulli(x / 2)
          x
        end

        is  = IS()
        smc = SMC()
        pg  = PG(10)

        res_is = sample(test(), is, 10000)
        res_smc = sample(test(), smc, 1000)
        res_pg = sample(test(), pg, 100)

        @test all(isone, res_is[:x].value)
        @test res_is.logevidence ≈ 2 * log(0.5)

        @test all(isone, res_smc[:x].value)
        @test res_smc.logevidence ≈ 2 * log(0.5)

        @test all(isone, res_pg[:x].value)
    end
    @testset "sample" begin
        alg = Gibbs(HMC(0.2, 3, :m), PG(10, :s))
        chn = sample(gdemo_default, alg, 1000);
    end
    @testset "vectorization @." begin
        @model vdemo1(x) = begin
            s ~ InverseGamma(2,3)
            m ~ Normal(0, sqrt(s))
            @. x ~ Normal(m, sqrt(s))
            return s, m
        end

        alg = HMC(0.01, 5)
        x = randn(100)
        res = sample(vdemo1(x), alg, 250)

        D = 2
        @model vdemo2(x) = begin
            μ ~ MvNormal(zeros(D), ones(D))
            @. x ~ MvNormal(μ, ones(D))
        end

        alg = HMC(0.01, 5)
        res = sample(vdemo2(randn(D,100)), alg, 250)

        # Vector assumptions
        N = 10
        setchunksize(N)
        alg = HMC(0.2, 4)

        @model vdemo3() = begin
            x = Vector{Real}(undef, N)
            for i = 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        t_loop = @elapsed res = sample(vdemo3(), alg, 1000)

        # Test for vectorize UnivariateDistribution
        @model vdemo4() = begin
          x = Vector{Real}(undef, N)
          @. x ~ Normal(0, 2)
        end

        t_vec = @elapsed res = sample(vdemo4(), alg, 1000)

        @model vdemo5() = begin
            x ~ MvNormal(zeros(N), 2 * ones(N))
        end

        t_mv = @elapsed res = sample(vdemo5(), alg, 1000)

        println("Time for")
        println("  Loop : $t_loop")
        println("  Vec  : $t_vec")
        println("  Mv   : $t_mv")

        # Transformed test
        @model vdemo6() = begin
            x = Vector{Real}(undef, N)
            @. x ~ InverseGamma(2, 3)
        end

        sample(vdemo6(), alg, 1000)

        N = 3
        @model vdemo7() = begin
            x = Array{Real}(undef, N, N)
            @. x ~ [InverseGamma(2, 3) for i in 1:N]
        end

        sample(vdemo7(), alg, 1000)
    end

    if VERSION >= v"1.1"
        """
        @testset "vectorization .~" begin
            @model vdemo1(x) = begin
                s ~ InverseGamma(2,3)
                m ~ Normal(0, sqrt(s))
                x .~ Normal(m, sqrt(s))
                return s, m
            end

            alg = HMC(0.01, 5)
            x = randn(100)
            res = sample(vdemo1(x), alg, 250)

            D = 2
            @model vdemo2(x) = begin
                μ ~ MvNormal(zeros(D), ones(D))
                x .~ MvNormal(μ, ones(D))
            end

            alg = HMC(0.01, 5)
            res = sample(vdemo2(randn(D,100)), alg, 250)

            # Vector assumptions
            N = 10
            setchunksize(N)
            alg = HMC(0.2, 4)

            @model vdemo3() = begin
                x = Vector{Real}(undef, N)
                for i = 1:N
                    x[i] ~ Normal(0, sqrt(4))
                end
            end

            t_loop = @elapsed res = sample(vdemo3(), alg, 1000)

            # Test for vectorize UnivariateDistribution
            @model vdemo4() = begin
            x = Vector{Real}(undef, N)
            x .~ Normal(0, 2)
            end

            t_vec = @elapsed res = sample(vdemo4(), alg, 1000)

            @model vdemo5() = begin
                x ~ MvNormal(zeros(N), 2 * ones(N))
            end

            t_mv = @elapsed res = sample(vdemo5(), alg, 1000)

            println("Time for")
            println("  Loop : \$t_loop")
            println("  Vec  : \$t_vec")
            println("  Mv   : \$t_mv")

            # Transformed test
            @model vdemo6() = begin
                x = Vector{Real}(undef, N)
                x .~ InverseGamma(2, 3)
            end

            sample(vdemo6(), alg, 1000)

            @model vdemo7() = begin
                x = Array{Real}(undef, N, N)
                x .~ [InverseGamma(2, 3) for i in 1:N]
            end
    
            sample(vdemo7(), alg, 1000)
        end
        """ |> Meta.parse |> eval
    end

    @testset "Type parameters" begin
        N = 10
        setchunksize(N)
        alg = HMC(0.01, 5)
        x = randn(1000)
        @model vdemo1(::Type{T}=Float64) where {T} = begin
            x = Vector{T}(undef, N)
            for i = 1:N
                x[i] ~ Normal(0, sqrt(4))
            end
        end

        t_loop = @elapsed res = sample(vdemo1(), alg, 250)
        t_loop = @elapsed res = sample(vdemo1(Float64), alg, 250)
        t_loop = @elapsed res = sample(vdemo1(T=Float64), alg, 250)

        @model vdemo2(::Type{T}=Float64) where {T <: Real} = begin
            x = Vector{T}(undef, N)
            @. x ~ Normal(0, 2)
        end

        t_vec = @elapsed res = sample(vdemo2(), alg, 250)
        t_vec = @elapsed res = sample(vdemo2(Float64), alg, 250)
        t_vec = @elapsed res = sample(vdemo2(T=Float64), alg, 250)

        @model vdemo3(::Type{TV}=Vector{Float64}) where {TV <: AbstractVector} = begin
            x = TV(undef, N)
            @. x ~ InverseGamma(2, 3)
        end

        sample(vdemo3(), alg, 250)
        sample(vdemo3(Vector{Float64}), alg, 250)
        sample(vdemo3(TV=Vector{Float64}), alg, 250)
    end
    @testset "split var string" begin
        var_str = "x"
        sym, inds = split_var_str(var_str)
        @test sym == "x"
        @test inds == Vector{String}[]

        var_str = "x[1,1][2,3]"
        sym, inds = split_var_str(var_str)
        @test sym == "x"
        @test inds[1] == ["1", "1"]
        @test inds[2] == ["2", "3"]

        var_str = "x[Colon(),1][2,Colon()]"
        sym, inds = split_var_str(var_str)
        @test sym == "x"
        @test inds[1] == ["Colon()", "1"]
        @test inds[2] == ["2", "Colon()"]

        var_str = "x[2:3,1][2,1:2]"
        sym, inds = split_var_str(var_str)
        @test sym == "x"
        @test inds[1] == ["2:3", "1"]
        @test inds[2] == ["2", "1:2"]

        var_str = "x[2:3,2:3][[1,2],[1,2]]"
        sym, inds = split_var_str(var_str)
        @test sym == "x"
        @test inds[1] == ["2:3", "2:3"]
        @test inds[2] == ["[1,2]", "[1,2]"]
    end
    @testset "user-defined variable name" begin
        @model f1() = begin
            x ~ NamedDist(Normal(), :y)
        end
        @model f2() = begin
            x ~ NamedDist(Normal(), Turing.@varname(y[2][:,1]))
        end
        @model f3() = begin
            x ~ NamedDist(Normal(), "y[1]")
        end
        vi1 = Turing.VarInfo(f1())
        vi2 = Turing.VarInfo(f2())
        vi3 = Turing.VarInfo(f3())
        @test haskey(vi1.metadata, :y)
        @test vi1.metadata.y.vns[1] == Turing.VarName{:y}("")
        @test haskey(vi2.metadata, :y)
        @test vi2.metadata.y.vns[1] == Turing.VarName{:y}("[2][Colon(),1]")
        @test haskey(vi3.metadata, :y)
        @test vi3.metadata.y.vns[1] == Turing.VarName{:y}("[1]")
    end
end
