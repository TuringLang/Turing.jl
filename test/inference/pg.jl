using Turing, Random, Test
using Turing.Inference: getspace
using AdvancedPS # TODO: maybe use import instead?
# import AdvancedPS

dir = splitdir(splitdir(pathof(Turing))[1])[1]
include(dir*"/test/test_utils/AllUtils.jl")


# relevant both for smc.jl and pg.jl
# mostly tests AdvancedPS functions and structs, but requires functions from inside inference/ eg PG()
@testset "particle container" begin
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
    Turing.@model empty_model() = begin x = 1; end
    sampler = Turing.Sampler(PG(5), empty_model())

    # Create particle container.
    logps = [0.0, -1.0, -2.0]
    particles = [Trace(fpc(logp), empty_model(), sampler, Turing.VarInfo()) for logp in logps]
    pc = ParticleContainer(particles)

    # Initial state.
    @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
    @test pc.logWs == zeros(3)
    @test getweights(pc) == fill(1/3, 3)
    @test all(getweight(pc, i) == 1/3 for i in 1:3)
    @test logZ(pc) ≈ log(3)
    @test effectiveSampleSize(pc) == 3

    # Reweight particles.
    reweight!(pc)
    @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
    @test pc.logWs == logps
    @test getweights(pc) ≈ exp.(logps) ./ sum(exp, logps)
    @test all(getweight(pc, i) ≈ exp(logps[i]) / sum(exp, logps) for i in 1:3)
    @test logZ(pc) == log(sum(exp, logps))

    # Reweight particles.
    reweight!(pc)
    @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
    @test pc.logWs == 2 .* logps
    @test getweights(pc) == exp.(2 .* logps) ./ sum(exp, 2 .* logps)
    @test all(getweight(pc, i) ≈ exp(2 * logps[i]) / sum(exp, 2 .* logps) for i in 1:3)
    @test logZ(pc) == log(sum(exp, 2 .* logps))

    # Resample and propagate particles.
    resample_propagate!(pc, resample_systematic)
    @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
    @test pc.logWs == zeros(3)
    @test getweights(pc) == fill(1/3, 3)
    @test all(getweight(pc, i) == 1/3 for i in 1:3)
    @test logZ(pc) ≈ log(3)
    @test effectiveSampleSize(pc) == 3

    # Reweight particles.
    reweight!(pc)
    @test all(iszero(getlogp(particle.vi)) for particle in pc.vals)
    @test pc.logWs ⊆ logps
    @test getweights(pc) == exp.(pc.logWs) ./ sum(exp, pc.logWs)
    @test all(getweight(pc, i) ≈ exp(pc.logWs[i]) / sum(exp, pc.logWs) for i in 1:3)
    @test logZ(pc) == log(sum(exp, pc.logWs))

    # Increase unnormalized logarithmic weights.
    logws = copy(pc.logWs)
    increase_logweight!(pc, 2, 1.41)
    @test pc.logWs == logws + [0, 1.41, 0]

    # Reset unnormalized logarithmic weights.
    logws = pc.logWs
    reset_logweights!(pc)
    @test pc.logWs === logws
    @test all(iszero, pc.logWs)
end

@testset "PG" begin
    @turing_testset "constructor" begin
        s = PG(10)
        @test s.nparticles == 10
        @test s.resampler == AdvancedPS.ResampleWithESSThreshold()
        @test getspace(s) === ()

        s = PG(20, :x)
        @test s.nparticles == 20
        @test s.resampler == AdvancedPS.ResampleWithESSThreshold()
        @test getspace(s) === (:x,)

        s = PG(30, (:x,))
        @test s.nparticles == 30
        @test s.resampler == AdvancedPS.ResampleWithESSThreshold()
        @test getspace(s) === (:x,)

        s = PG(40, :x, :y)
        @test s.nparticles == 40
        @test s.resampler == AdvancedPS.ResampleWithESSThreshold()
        @test getspace(s) === (:x, :y)

        s = PG(50, (:x, :y))
        @test s.nparticles == 50
        @test s.resampler == AdvancedPS.ResampleWithESSThreshold()
        @test getspace(s) === (:x, :y)

        s = PG(60, 0.6)
        @test s.nparticles == 60
        @test s.resampler === AdvancedPS.ResampleWithESSThreshold(AdvancedPS.resample_systematic, 0.6)
        @test getspace(s) === ()

        s = PG(70, 0.6, (:x,))
        @test s.nparticles == 70
        @test s.resampler === AdvancedPS.ResampleWithESSThreshold(AdvancedPS.resample_systematic, 0.6)
        @test getspace(s) === (:x,)

        s = PG(80, AdvancedPS.resample_multinomial, 0.6)
        @test s.nparticles == 80
        @test s.resampler === AdvancedPS.ResampleWithESSThreshold(AdvancedPS.resample_multinomial, 0.6)
        @test getspace(s) === ()

        s = PG(90, AdvancedPS.resample_multinomial, 0.6, (:x,))
        @test s.nparticles == 90
        @test s.resampler === AdvancedPS.ResampleWithESSThreshold(AdvancedPS.resample_multinomial, 0.6)
        @test getspace(s) === (:x,)

        s = PG(100, AdvancedPS.resample_systematic)
        @test s.nparticles == 100
        @test s.resampler === AdvancedPS.resample_systematic
        @test getspace(s) === ()

        s = PG(110, AdvancedPS.resample_systematic, (:x,))
        @test s.nparticles == 110
        @test s.resampler === AdvancedPS.resample_systematic
        @test getspace(s) === (:x,)
    end

    @turing_testset "logevidence" begin
        Random.seed!(100)

        @model function test()
            a ~ Normal(0, 1)
            x ~ Bernoulli(1)
            b ~ Gamma(2, 3)
            1 ~ Bernoulli(x / 2)
            c ~ Beta()
            0 ~ Bernoulli(x / 2)
            x
        end

        chains_pg = sample(test(), PG(10), 100)

        @test all(isone, chains_pg[:x].value)
        @test chains_pg.logevidence ≈ -2 * log(2) atol = 0.01
    end
end