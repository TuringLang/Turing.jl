using Turing
using LogDensityProblems
using Test

struct BadModel end

function LogDensityProblems.logdensity(::BadModel, x)
    error("boom")
end

@testset "SafeLogDensity local test" begin
    wrapped = Turing.SafeLogDensity(BadModel())
    val = LogDensityProblems.logdensity(wrapped, [1.0])
    @test val == -Inf
end