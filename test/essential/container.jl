module ContainerTests

using Distributions: Bernoulli, Beta, Gamma, Normal
using DynamicPPL: DynamicPPL, @model
using Random: Xoshiro
using Test: @test, @testset
using Turing
using Turing.Inference:
    Particle, TracedRNG, particle_varinfo, advance!, fork, set_step!, refresh!

@testset "container.jl" begin
    @model function test()
        a ~ Normal(0, 1)
        x ~ Bernoulli(1)
        b ~ Gamma(2, 3)
        1 ~ Bernoulli(x / 2)
        c ~ Beta()
        0 ~ Bernoulli(x / 2)
        return x
    end

    @testset "advance!" begin
        # `x ~ Bernoulli(1)` forces `x = 1`, so the first observe is `1 ~ Bernoulli(0.5)`.
        particle = Particle(test(), particle_varinfo(), TracedRNG(Xoshiro(23)))
        @test advance!(particle, false) ≈ -log(2)
        @test advance!(particle, false) ≈ -log(2)     # `0 ~ Bernoulli(0.5)`
        @test advance!(particle, false) === nothing    # model finished
    end

    @testset "fork" begin
        particle = Particle(test(), particle_varinfo(), TracedRNG(Xoshiro(23)))
        advance!(particle, false)
        child = fork(particle, Xoshiro(1))
        # Independent continuations: advancing one does not touch the other.
        @test advance!(child, false) ≈ -log(2)
        @test particle.varinfo !== child.varinfo
        @test advance!(particle, false) ≈ -log(2)
    end

    @testset "rng replay" begin
        @model function normal()
            a ~ Normal(0, 1)
            3 ~ Normal(a, 2)
            b ~ Normal(a, 1)
            1.5 ~ Normal(b, 2)
            return a, b
        end

        # Run a particle to completion, then replay it from its recorded seeds (as the
        # reference trajectory of a conditional sweep does) and check it regenerates exactly.
        # Replay relies on each step using a distinct seed, so we refresh before every step
        # exactly as the sweep's no-resample path does.
        particle = Particle(normal(), particle_varinfo(), TracedRNG(Xoshiro(23)))
        while (refresh!(particle.rng); advance!(particle, false)) !== nothing
        end
        values = DynamicPPL.get_raw_values(particle.varinfo)

        reference = Particle(
            normal(), particle_varinfo(), set_step!(deepcopy(particle.rng), 1)
        )
        while advance!(reference, true) !== nothing
        end
        @test DynamicPPL.get_raw_values(reference.varinfo) == values
    end
end

end
