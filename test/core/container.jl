using Turing, Random
using Turing: ParticleContainer, weights, resample!,
    effectiveSampleSize, Trace, current_trace, VarName,
    Sampler, consume, produce, copy, fork
using Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "container.jl" begin
    @turing_testset "copy particle container" begin
        pc = ParticleContainer{Trace}(x -> x * x)
        newpc = copy(pc)

        @test newpc.logE        == pc.logE
        @test newpc.logWs       == pc.logWs
        @test newpc.conditional == pc.conditional
        @test newpc.n_consume   == pc.n_consume
    end
    @turing_testset "particle container" begin
        n = Ref(0)

        alg = PG(5, 1)
        spl = Turing.Sampler(alg)
        dist = Normal(0, 1)

        function fpc()
            t = TArray(Float64, 1);
            t[1] = 0;
            while true
                ct = current_trace()
                vn = VarName(gensym(), :x, "[$n]", 1)
                Turing.assume(spl, dist, vn, ct.vi); n[] += 1;
                produce(0)
                vn = VarName(gensym(), :x, "[$n]", 1)
                Turing.assume(spl, dist, vn, ct.vi); n[] += 1;
                t[1] = 1 + t[1]
            end
        end

        pc = ParticleContainer{Trace}(fpc)

        push!(pc, Trace(pc.model))
        push!(pc, Trace(pc.model))
        push!(pc, Trace(pc.model))

        Base.@assert weights(pc)[1] == [1/3, 1/3, 1/3]
        Base.@assert weights(pc)[2] ≈ log(3)
        Base.@assert pc.logE ≈ log(1)

        Base.@assert consume(pc) == log(1)

        resample!(pc)
        Base.@assert pc.num_particles == length(pc)
        Base.@assert weights(pc)[1] == [1/3, 1/3, 1/3]
        Base.@assert weights(pc)[2] ≈ log(3)
        Base.@assert pc.logE ≈ log(1)
        Base.@assert effectiveSampleSize(pc) == 3

        Base.@assert consume(pc) ≈ log(1)
        resample!(pc)
        Base.@assert consume(pc) ≈ log(1)
    end
    @turing_testset "trace" begin
        n = Ref(0)

        alg = PG(5, 1)
        spl = Turing.Sampler(alg)
        dist = Normal(0, 1)
        function f2()
            t = TArray(Int, 1);
            t[1] = 0;
            while true
                ct = current_trace()
                vn = VarName(gensym(), :x, "[$n]", 1)
                Turing.assume(spl, dist, vn, ct.vi); n[] += 1;
                produce(t[1]);
                vn = VarName(gensym(), :x, "[$n]", 1)
                Turing.assume(spl, dist, vn, ct.vi); n[] += 1;
                t[1] = 1 + t[1]
            end
        end

        # Test task copy version of trace
        tr = Trace(f2)

        consume(tr); consume(tr)
        a = fork(tr);
        consume(a); consume(a)

        Base.@assert consume(tr) == 2
        Base.@assert consume(a) == 4
    end
end
