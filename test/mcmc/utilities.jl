module MCMCUtilitiesTests

using ..Models: gdemo_default
using FlexiChains: FlexiChains
using Test: @test, @testset
using Turing

@testset "Timer" begin
    chain = sample(gdemo_default, MH(), 1000)
    @test FlexiChains.sampling_time(chain) isa Vector{Float64}
    @test only(FlexiChains.sampling_time(chain)) > 0.0
end

end
