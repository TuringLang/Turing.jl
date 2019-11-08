using Turing, Random, Test
using IterTools

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "dpp.jl" begin
    @turing_testset "DeterminantalPointProcess" begin
        n = 10
        rng = MersenneTwister(1)
        x = rand(rng, n, 5)

        # kernel
        L = x * x'

        # determinantal point process
        d = DeterminantalPointProcess(L)

        # helper function
        function onehot(d::Int, z::AbstractVector{Int})
            @assert minimum(z) >= 1
            @assert maximum(z) <= d
            x = falses(d)
            @inbounds x[z] .= true
            return x
        end

        lpmfs = Vector{Float64}()
        for z in subsets(1:n)
            x = onehot(n, z)
            lp = logpdf(d, x)
            push!(lpmfs, lp)
        end

        @test sum(lpmfs) ≈ 0
        @test all(lpmfs .<= 0)
    end
end
