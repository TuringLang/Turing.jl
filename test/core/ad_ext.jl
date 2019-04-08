using Distributions, Tracker, PDMats, LinearAlgebra
using Turing: mvnormlogpdf
using Test

include("../test_utils/AllUtils.jl")

@testset "ad_ext.jl" begin
    @show "start..."
    @turing_testset "`MvNormal` and `Tracker` compatibility" begin
        Random.seed!(0);
        A = randn(3, 3);
        M = A * A' + I;
        u, x = randn(3), randn(3);

        y, back = Tracker.forward((u, M, x)->logpdf(MvNormal(u, PDMat(M)), x), u, M, x)

        @test typeof(y) == Tracker.TrackedReal{Float64}
        @show y
        @show back(1)

        y, back = Tracker.forward((u, M, x)->mvnormlogpdf(u, M, x), u, M, x)
        @show y
        @show back(1)
    end
end

