module MCMCUtilitiesTests

using ..Models: gdemo_default
using Test: @test, @testset
using Turing

@testset "Timer" begin
    chain = sample(gdemo_default, MH(), 1000)

    @test chain.info.start_time isa Float64
    @test chain.info.stop_time isa Float64
    @test chain.info.start_time ≤ chain.info.stop_time
end

end
