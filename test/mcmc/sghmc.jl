@testset "Testing sghmc.jl with $adbackend" for adbackend in (AutoForwardDiff(; chunksize=0), AutoReverseDiff(false))
    @turing_testset "sghmc constructor" begin
        alg = SGHMC(; learning_rate=0.01, momentum_decay=0.1, adtype=adbackend)
        @test alg isa SGHMC
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGHMC}

        alg = SGHMC(:m; learning_rate=0.01, momentum_decay=0.1, adtype=adbackend)
        @test alg isa SGHMC
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGHMC}

        alg = SGHMC(:s; learning_rate=0.01, momentum_decay=0.1, adtype=adbackend)
        @test alg isa SGHMC
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGHMC}
    end
    @numerical_testset "sghmc inference" begin
        rng = StableRNG(123)

        alg = SGHMC(; learning_rate=0.02, momentum_decay=0.5, adtype=adbackend)
        chain = sample(rng, gdemo_default, alg, 10_000)
        check_gdemo(chain, atol=0.1)
    end
end

@testset "Testing sgld.jl with $adbackend" for adbackend in (AutoForwardDiff(; chunksize=0), AutoReverseDiff(false))
    @turing_testset "sgld constructor" begin
        alg = SGLD(; stepsize=PolynomialStepsize(0.25), adtype=adbackend)
        @test alg isa SGLD
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGLD}

        alg = SGLD(:m; stepsize=PolynomialStepsize(0.25), adtype=adbackend)
        @test alg isa SGLD
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGLD}

        alg = SGLD(:s; stepsize=PolynomialStepsize(0.25), adtype=adbackend)
        @test alg isa SGLD
        sampler = Turing.Sampler(alg)
        @test sampler isa Turing.Sampler{<:SGLD}
    end
    @numerical_testset "sgld inference" begin
        rng = StableRNG(1)

        chain = sample(rng, gdemo_default, SGLD(; stepsize=PolynomialStepsize(0.5)), 20_000)
        check_gdemo(chain, atol=0.2)

        # Weight samples by step sizes (cf section 4.2 in the paper by Welling and Teh)
        v = get(chain, [:SGLD_stepsize, :s, :m])
        s_weighted = dot(v.SGLD_stepsize, v.s) / sum(v.SGLD_stepsize)
        m_weighted = dot(v.SGLD_stepsize, v.m) / sum(v.SGLD_stepsize)
        @test s_weighted ≈ 49 / 24 atol = 0.2
        @test m_weighted ≈ 7 / 6 atol = 0.2
    end
end
