using Distributions, Tracker, PDMats, LinearAlgebra
using Turing
using Test

include("../test_utils/AllUtils.jl")

@testset "ad_ext.jl" begin
    @turing_testset "`MvNormal` and `Tracker` compatibility" begin
        Random.seed!(0);
        A = randn(3, 3);
        B = A * A' + I;
        x, m = randn(3), randn(3), B;

        y, back = Tracker.forward((x, m, B)->logpdf(MvNormal(m, PDMat(B)), x), x, m, B)

        @test typeof(y) == Tracker.TrackedReal{Float64}
    end
end

