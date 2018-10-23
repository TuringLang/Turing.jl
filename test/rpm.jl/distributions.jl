using Turing
using Test

@testset "Representations" begin
    d = StickBreakingProcess(DirichletProcess(1.0))
    @test minimum(d) == 0
    @test maximum(d) == 1

    d = SizeBiasedSamplingProcess(DirichletProcess(1.0), 2.0)
    @test minimum(d) == 0
    @test maximum(d) == 2

    d = ChineseRestaurantProcess(DirichletProcess(1.0), [2, 1])
    @test minimum(d) == 1
    @test maximum(d) == 3
end

@testset "Dirichlet Process" begin

    α = 0.1
    N = 10_000

    # test SB representation
    d = StickBreakingProcess(DirichletProcess(α))
    Ev = mapreduce(_ -> rand(d), +, 1:N) / N
    @test Ev ≈ mean(Beta(1, α)) atol=0.05

    # test SBS representation
    d = SizeBiasedSamplingProcess(DirichletProcess(α), 2.0)
    Ej = mapreduce(_ -> rand(d), +, 1:N) / N
    @test Ej ≈ mean(Beta(1, α)) * 2 atol=0.05

    # test CRP representation
    d = ChineseRestaurantProcess(DirichletProcess(α), [2, 1])
    ks = map(_ -> rand(d), 1:N)
    c = map(k -> sum(ks .== k), support(d))
    p = c ./ sum(c)

    q = [2, 1, α] ./ (2 + α)
    q ./= sum(q)

    @test p[1] ≈ q[1] atol=0.1
    @test p[2] ≈ q[2] atol=0.1
    @test p[3] ≈ q[3] atol=0.1
end

@testset "Pitman-Yor Process" begin

    a = 0.5
    θ = 0.1
    t = 2
    N = 10_000

    # test SB representation
    d = StickBreakingProcess(PitmanYorProcess(a, θ, t))
    Ev = mapreduce(_ -> rand(d), +, 1:N) / N
    @test Ev ≈ mean(Beta(1 - a, θ + t*a)) atol=0.05

    # test SBS representation
    d = SizeBiasedSamplingProcess(PitmanYorProcess(a, θ, t), 2.0)
    Ej = mapreduce(_ -> rand(d), +, 1:N) / N
    @test Ej ≈ mean(Beta(1 - a, θ + t*a)) * 2 atol=0.05

    # test CRP representation
    d = ChineseRestaurantProcess(PitmanYorProcess(a, θ, t), [2, 1])
    ks = map(_ -> rand(d), 1:N)
    c = map(k -> sum(ks .== k), support(d))
    p = c ./ sum(c)

    q = [2 - a, 1 - a, θ + a*t] ./ (3 + θ)
    q ./= sum(q)

    @test p[1] ≈ q[1] atol=0.1
    @test p[2] ≈ q[2] atol=0.1
    @test p[3] ≈ q[3] atol=0.1
end
