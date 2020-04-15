using Turing, Random, Test
using StatsFuns

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@turing_testset "smc.jl" begin
  @model normal() = begin
    a ~ Normal(4,5)
    3 ~ Normal(a,2)
    b ~ Normal(a,1)
    1.5 ~ Normal(b,2)
    a, b
  end

  tested = sample(normal(), SMC(), 100);

  # failing test
  @model fail_smc() = begin
    a ~ Normal(4,5)
    3 ~ Normal(a,2)
    b ~ Normal(a,1)
    if a >= 4.0
      1.5 ~ Normal(b,2)
    end
    a, b
  end

  @test_throws ErrorException sample(fail_smc(), SMC(), 100)
end
