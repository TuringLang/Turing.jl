using Turing, Random
using Turing: ParticleContainer, getweights, resample!,
    effectiveSampleSize, Trace, current_trace, VarName,
    Sampler, consume, produce, copy, fork
using Turing.Core: logZ
using Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "container.jl" begin
    @turing_testset "copy particle container" begin
        pc = ParticleContainer(x -> x * x, Trace[])
        newpc = copy(pc)

        @test newpc.logE        == pc.logE
        @test newpc.logWs       == pc.logWs
        @test newpc.n_consume   == pc.n_consume
        @test typeof(pc) === typeof(newpc)
    end
    @turing_testset "particle container" begin
        n = Ref(0)

        alg = PG(5)
        spl = Turing.Sampler(alg, empty_model())
        dist = Normal(0, 1)

        function fpc()
            t = TArray(Float64, 1);
            t[1] = 0;
            while true
                ct = current_trace()
                vn = @varname x[n]
                Turing.assume(spl, dist, vn, ct.vi); n[] += 1;
                produce(0)
                vn = @varname x[n]
                Turing.assume(spl, dist, vn, ct.vi); n[] += 1;
                t[1] = 1 + t[1]
            end
        end

        modelgen = Turing.ModelGen{()}(nothing, NamedTuple())
        model = Turing.Model(fpc, NamedTuple(), modelgen)
        particles = [Trace(fpc, model, spl, Turing.VarInfo()) for _ in 1:3]
        pc = ParticleContainer(fpc, particles)

        @test getweights(pc) == [1/3, 1/3, 1/3]
        @test logZ(pc) ≈ log(3)
        @test pc.logE ≈ log(1)

        @test consume(pc) == log(1)

        resample!(pc)
        @test getweights(pc) == [1/3, 1/3, 1/3]
        @test logZ(pc) ≈ log(3)
        @test pc.logE ≈ log(1)
        @test effectiveSampleSize(pc) == 3

        @test consume(pc) ≈ log(1)
        resample!(pc)
        @test consume(pc) ≈ log(1)
    end
    @turing_testset "trace" begin
        n = Ref(0)

        alg = PG(5)
        spl = Turing.Sampler(alg, empty_model())
        dist = Normal(0, 1)
        function f2()
            t = TArray(Int, 1);
            t[1] = 0;
            while true
                ct = current_trace()
                vn = @varname x[n]
                Turing.assume(spl, dist, vn, ct.vi); n[] += 1;
                produce(t[1]);
                vn = @varname x[n]
                Turing.assume(spl, dist, vn, ct.vi); n[] += 1;
                t[1] = 1 + t[1]
            end
        end

        # Test task copy version of trace
        modelgen = Turing.ModelGen{()}(nothing, NamedTuple())
        model = Turing.Model(f2, NamedTuple(), modelgen)
        tr = Trace(f2, model, spl, Turing.VarInfo())

        consume(tr); consume(tr)
        a = fork(tr);
        consume(a); consume(a)

        @test consume(tr) == 2
        @test consume(a) == 4
    end
end
