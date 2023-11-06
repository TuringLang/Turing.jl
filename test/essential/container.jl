@testset "container.jl" begin
    @turing_testset "constructor" begin

      @model function test()
          a ~ Normal(0, 1)
          x ~ Bernoulli(1)
          b ~ Gamma(2, 3)
          1 ~ Bernoulli(x / 2)
          c ~ Beta()
          0 ~ Bernoulli(x / 2)
          x
      end

      vi = DynamicPPL.VarInfo()
      sampler = Sampler(PG(10))
      model = test()
      trace = Trace(model, sampler, vi, TracedRNG())

    @test haskey(trace.model.ctask.task.storage, :__trace)

    res = AdvancedPS.advance!(trace, false)
    @test DynamicPPL.get_num_produce(trace.model.f.varinfo) == 1
    @test res ≈ -log(2)

    newtrace = copy(trace)
    res2 = AdvancedPS.advance!(trace)
    @test DynamicPPL.get_num_produce(trace.model.f.varinfo) == 2
    @test DynamicPPL.get_num_produce(newtrace.model.f.varinfo) == 1
    end
end
