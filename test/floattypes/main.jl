# Script to use for testing promotion of log-prob types. Since this relies on compile-time
# preferences, it's hard to run this within the usual CI setup.
#
# Usage:
#    julia --project=. main.jl setup f32  # Sets the preference
#    julia --project=. main.jl run f32    # Checks that the preference is respected
#
# and this should be looped over for `f64`, `f32`, `f16`, and `min`.

using DynamicPPL, Turing, Test

function floattypestr_to_type(floattypestr)
    if floattypestr == "f64"
        return Float64
    elseif floattypestr == "f32"
        return Float32
    elseif floattypestr == "f16"
        return Float16
    elseif floattypestr == "min"
        return DynamicPPL.NoLogProb
    else
        error("Invalid float type: $floattypestr")
    end
end

function setup(floattypestr)
    T = floattypestr_to_type(floattypestr)
    return set_logprob_type!(T)
end

function test_with_type(::Type{T}) where {T}
    @testset "Testing with type $T" begin
        @model f() = x ~ Normal(T(0), T(1))

        # optimisation
        mr = maximum_a_posteriori(f())
        @test mr.params[@varname(x)] isa T
        @test mr.lp isa T
        @test eltype(mr.optim_result.u) == T

        # MH and Prior work
        @testset "$spl" for spl in (MH(), Prior(), Gibbs(:x => MH()))
            chn = sample(f(), spl, 10; progress=false)
            @test eltype(chn[@varname(x)]) == T
        end

        # Known failures (note MH([1.0;;]) is really externalsampler with AdvancedMH)
        if T != Float64
            @testset "$spl" for spl in (ESS(), MH([1.0;;]))
                chn = sample(f(), spl, 10; progress=false)
                @test_broken eltype(chn[@varname(x)]) == T
            end

            # AdvancedHMC straight up errors :-(
            @testset "$spl" for spl in (HMC(0.1, 5), NUTS())
                @test_throws Exception sample(f(), spl, 10; progress=false)
            end
        end
    end
end

function run(floattypestr)
    T = floattypestr_to_type(floattypestr)
    if T == DynamicPPL.NoLogProb
        @test DynamicPPL.LogProbType === DynamicPPL.NoLogProb
        # all higher types should cause promotion to those types
        test_with_type(Float16)
        test_with_type(Float32)
        test_with_type(Float64)
    else
        @test DynamicPPL.LogProbType === T
        test_with_type(T)
    end
end

if length(ARGS) != 2 ||
    !(ARGS[1] in ["setup", "run"]) ||
    !(ARGS[2] in ["f64", "f32", "f16", "min"])
    println("Usage: julia --project=. main.jl <setup|run> <f64|f32|f16|min>")
    exit(1)
end

mode = ARGS[1]
floattypestr = ARGS[2]
if mode == "setup"
    setup(floattypestr)
elseif mode == "run"
    run(floattypestr)
end
