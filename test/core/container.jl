using Turing, Random
using Turing: ParticleContainer, getweights, resample!,
    effectiveSampleSize, Trace, current_trace, VarName, VarInfo,
    Sampler, consume, produce, copy, fork, getlogp
using Turing.Core: logZ, propagate!
using Test

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")

@testset "container.jl" begin
    @turing_testset "copy particle container" begin
        pc = ParticleContainer(Trace[])
        newpc = copy(pc)

        @test newpc.logWs == pc.logWs
        @test typeof(pc) === typeof(newpc)
    end
    @turing_testset "particle container" begin
        # Create a resumable function that always yields `logp`.
        function fpc(logp)
            f = let logp = logp
                () -> begin
                    while true
                        produce(logp)
                    end
                end
            end
            return f
        end

        # Dummy sampler that is not actually used.
        sampler = Sampler(PG(5), empty_model())

        # Create particle container.
        logps = [0.0, -1.0, -2.0]
        particles = [Trace(fpc(logp), empty_model(), sampler, VarInfo()) for logp in logps]
        pc = ParticleContainer(particles)

        # Initial state.
        @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
        @test pc.logWs == zeros(3)
        @test getweights(pc) == fill(1/3, 3)
        @test iszero(logZ(pc))
        @test effectiveSampleSize(pc) == 3

        # Propagate particles.
        propagate!(pc)
        @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
        @test pc.logWs == logps
        @test getweights(pc) ≈ exp.(logps) ./ sum(exp, logps)
        @test logZ(pc) == log(sum(exp, logps)) - log(3)

        # Propagate particles.
        propagate!(pc)
        @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
        @test pc.logWs == 2 .* logps
        @test getweights(pc) == exp.(2 .* logps) ./ sum(exp, 2 .* logps)
        @test logZ(pc) == log(sum(exp, 2 .* logps)) - log(3)

        # Resample particles.
        resample!(pc)
        @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
        @test pc.logWs == zeros(3)
        @test getweights(pc) == fill(1/3, 3)
        @test iszero(logZ(pc))
        @test effectiveSampleSize(pc) == 3

        # Propagate particles.
        propagate!(pc)
        @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
        @test pc.logWs ⊆ logps
        @test getweights(pc) == exp.(pc.logWs) ./ sum(exp, pc.logWs)
        @test logZ(pc) == log(sum(exp, pc.logWs)) - log(3)
    end
    @turing_testset "trace" begin
        n = Ref(0)

        alg = PG(5)
        spl = Sampler(alg, empty_model())
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
        tr = Trace(f2, empty_model(), spl, VarInfo())

        consume(tr); consume(tr)
        a = fork(tr);
        consume(a); consume(a)

        @test consume(tr) == 2
        @test consume(a) == 4
    end
end
