# Script to use for testing promotion of log-prob types. Since this relies on compile-time
# preferences, it's hard to run this within the usual CI setup.
#
# Usage:
#    julia --project=. main.jl setup f32  # Sets the preference
#    julia --project=. main.jl run f32    # Checks that the preference is respected
#
# and this should be looped over for `f64`, `f32`, `f16`, and `min`.

using DynamicPPL, LogDensityProblems, ForwardDiff, Distributions, ADTypes, Test

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
    return DynamicPPL.set_logprob_type!(T)
end

function test_with_type(::Type{T}) where {T}
    @testset "Testing with type $T" begin
        @model f() = x ~ Normal(T(0), T(1))
        model = f()
        vnt = rand(model)
        @test vnt[@varname(x)] isa T
        lj = (@inferred logjoint(f(), (; x=T(0.0))))
        @test lj isa T
        ldf = LogDensityFunction(
            f(), getlogjoint_internal, LinkAll(); adtype=AutoForwardDiff()
        )
        @test rand(ldf) isa AbstractVector{T}
        lp = (@inferred LogDensityProblems.logdensity(ldf, [T(0)]))
        @test lp isa T
        @test lp ≈ logpdf(Normal(T(0), T(1)), T(0))
        lp_and_grad = (@inferred LogDensityProblems.logdensity_and_gradient(ldf, [T(0)]))
        @test first(lp_and_grad) isa T
        @test last(lp_and_grad) isa AbstractVector{T}
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
