using Turing, Random, Bijectors
using Distributions
using Turing.RandomVariables
using Test

@testset "Common/Runners" begin
    @turing_testset "ComputeLogJointDensity" begin

        # Test assume with single random variable (unconstrained)
        vi = VarInfo()
        vn = VarName(gensym(), :x, "", 1)
        dist = Normal(0,1)
        r = rand(dist)
        gid = Turing.Selector()
        runner = Turing.ComputeLogJointDensity()

        push!(vi, vn, r, dist, gid)
        @test Turing.assume(runner, dist, vn, vi) == (r, logpdf(dist, r))

        # Test assume with single random variable (constrained)
        vi = VarInfo()
        vn = VarName(gensym(), :x, "", 1)
        dist = Truncated(Normal(0,1), 0, Inf)
        r = link(dist, rand(dist))
        gid = Turing.Selector()
        runner = Turing.ComputeLogJointDensity()

        push!(vi, vn, r, dist, gid)
        Turing.RandomVariables.settrans!(vi, true, vn)
        r_ = invlink(dist, r)
        @test Turing.assume(runner, dist, vn, vi) == (r_, logpdf_with_trans(dist, r_, true))

        # Test observe with single random variable

    end
end
