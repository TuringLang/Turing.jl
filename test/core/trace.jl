using AdvancedPS
using Random
using Test

@testset "trace.jl" begin
        n = Ref(0)

        alg = PG(5)
        spl = Turing.Inference.Sampler(alg, empty_model())
        dist = Normal(0, 1)
        function f2()
            t = TArray(Int, 1);
            t[1] = 0;
            while true
                ct = current_trace()
                vn = @varname x[n]
                assume(Random.GLOBAL_RNG, spl, dist, vn, ct.vi)
                n[] += 1
                produce(t[1])
                vn = @varname x[n]
                assume(Random.GLOBAL_RNG, spl, dist, vn, ct.vi)
                n[] += 1
                t[1] = 1 + t[1]
            end
        end

        # Test task copy version of trace
        tr = Trace(f2, empty_model(), spl, Turing.VarInfo())

        consume(tr); consume(tr)
        a = fork(tr);
        consume(a); consume(a)

        @test consume(tr) == 2
        @test consume(a) == 4
    end