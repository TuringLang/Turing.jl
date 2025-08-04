module TuringADTests

using Turing
using DynamicPPL
using DynamicPPL.TestUtils: DEMO_MODELS
using DynamicPPL.TestUtils.AD: run_ad
using Random: Random
using StableRNGs: StableRNG
using Test
using ..Models: gdemo_default
import ForwardDiff, ReverseDiff

# Detect if prerelease version, if so, we skip some tests
const IS_PRERELEASE = !isempty(VERSION.prerelease)
const INCLUDE_MOONCAKE = !IS_PRERELEASE

if INCLUDE_MOONCAKE
    import Pkg
    Pkg.add("Mooncake")
    using Mooncake: Mooncake
end

const INCLUDE_ENZYME = !IS_PRERELEASE

if INCLUDE_ENZYME
    import Pkg
    Pkg.add("Enzyme")
    using Enzyme: Enzyme
end

"""Element types that are always valid for a VarInfo regardless of ADType."""
const always_valid_eltypes = (AbstractFloat, AbstractIrrational, Integer, Rational)

"""A dictionary mapping ADTypes to the element types they use."""
eltypes_by_adtype = Dict{Type,Tuple}(
    AutoForwardDiff => (ForwardDiff.Dual,),
    AutoReverseDiff => (
        ReverseDiff.TrackedArray,
        ReverseDiff.TrackedMatrix,
        ReverseDiff.TrackedReal,
        ReverseDiff.TrackedStyle,
        ReverseDiff.TrackedType,
        ReverseDiff.TrackedVecOrMat,
        ReverseDiff.TrackedVector,
    ),
)
if INCLUDE_MOONCAKE
    eltypes_by_adtype[AutoMooncake] = (Mooncake.CoDual,)
end
if INCLUDE_ENZYME
    eltypes_by_adtype[AutoEnzyme] = ()
end

"""
    AbstractWrongADBackendError

An abstract error thrown when we seem to be using a different AD backend than expected.
"""
abstract type AbstractWrongADBackendError <: Exception end

"""
    WrongADBackendError

An error thrown when we seem to be using a different AD backend than expected.
"""
struct WrongADBackendError <: AbstractWrongADBackendError
    actual_adtype::Type
    expected_adtype::Type
end

function Base.showerror(io::IO, e::WrongADBackendError)
    return print(
        io, "Expected to use $(e.expected_adtype), but using $(e.actual_adtype) instead."
    )
end

"""
    IncompatibleADTypeError

An error thrown when an element type is encountered that is unexpected for the given ADType.
"""
struct IncompatibleADTypeError <: AbstractWrongADBackendError
    valtype::Type
    adtype::Type
end

function Base.showerror(io::IO, e::IncompatibleADTypeError)
    return print(
        io,
        "Incompatible ADType: Did not expect element of type $(e.valtype) with $(e.adtype)",
    )
end

"""
    ADTypeCheckContext{ADType,ChildContext}

A context for checking that the expected ADType is being used.

Evaluating a model with this context will check that the types of values in a `VarInfo` are
compatible with the ADType of the context. If the check fails, an `IncompatibleADTypeError`
is thrown.

For instance, evaluating a model with
`ADTypeCheckContext(AutoForwardDiff(), child_context)`
would throw an error if within the model a type associated with e.g. ReverseDiff was
encountered.

"""
struct ADTypeCheckContext{ADType,ChildContext<:DynamicPPL.AbstractContext} <:
       DynamicPPL.AbstractContext
    child::ChildContext

    function ADTypeCheckContext(adbackend, child)
        adtype = adbackend isa Type ? adbackend : typeof(adbackend)
        if !any(adtype <: k for k in keys(eltypes_by_adtype))
            throw(ArgumentError("Unsupported ADType: $adtype"))
        end
        return new{adtype,typeof(child)}(child)
    end
end

adtype(_::ADTypeCheckContext{ADType}) where {ADType} = ADType

DynamicPPL.NodeTrait(::ADTypeCheckContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(c::ADTypeCheckContext) = c.child
function DynamicPPL.setchildcontext(c::ADTypeCheckContext, child)
    return ADTypeCheckContext(adtype(c), child)
end

"""
    valid_eltypes(context::ADTypeCheckContext)

Return the element types that are valid for the ADType of `context` as a tuple.
"""
function valid_eltypes(context::ADTypeCheckContext)
    context_at = adtype(context)
    for at in keys(eltypes_by_adtype)
        if context_at <: at
            return (eltypes_by_adtype[at]..., always_valid_eltypes...)
        end
    end
    # This should never be reached due to the check in the inner constructor.
    throw(ArgumentError("Unsupported ADType: $(adtype(context))"))
end

"""
    check_adtype(context::ADTypeCheckContext, vi::DynamicPPL.VarInfo)

Check that the element types in `vi` are compatible with the ADType of `context`.

Throw an `IncompatibleADTypeError` if an incompatible element type is encountered.
"""
function check_adtype(context::ADTypeCheckContext, vi::DynamicPPL.AbstractVarInfo)
    valids = valid_eltypes(context)
    for val in vi[:]
        valtype = typeof(val)
        if !any(valtype .<: valids)
            throw(IncompatibleADTypeError(valtype, adtype(context)))
        end
    end
    return nothing
end

# A bunch of tilde_assume/tilde_observe methods that just call the same method on the child
# context, and then call check_adtype on the result before returning the results from the
# child context.

function DynamicPPL.tilde_assume(context::ADTypeCheckContext, right, vn, vi)
    value, vi = DynamicPPL.tilde_assume(DynamicPPL.childcontext(context), right, vn, vi)
    check_adtype(context, vi)
    return value, vi
end

function DynamicPPL.tilde_assume(
    rng::Random.AbstractRNG, context::ADTypeCheckContext, sampler, right, vn, vi
)
    value, vi = DynamicPPL.tilde_assume(
        rng, DynamicPPL.childcontext(context), sampler, right, vn, vi
    )
    check_adtype(context, vi)
    return value, vi
end

function DynamicPPL.tilde_observe!!(context::ADTypeCheckContext, right, left, vi)
    left, vi = DynamicPPL.tilde_observe!!(DynamicPPL.childcontext(context), right, left, vi)
    check_adtype(context, vi)
    return left, vi
end

function DynamicPPL.tilde_observe!!(context::ADTypeCheckContext, sampler, right, left, vi)
    left, vi = DynamicPPL.tilde_observe!!(
        DynamicPPL.childcontext(context), sampler, right, left, vi
    )
    check_adtype(context, vi)
    return left, vi
end

"""
All the ADTypes on which we want to run the tests.
"""
ADTYPES = [AutoForwardDiff(), AutoReverseDiff(; compile=false)]
if INCLUDE_MOONCAKE
    push!(ADTYPES, AutoMooncake(; config=nothing))
end
if INCLUDE_ENZYME
    push!(
        ADTYPES,
        AutoEnzyme(;
            mode=Enzyme.set_runtime_activity(Enzyme.Forward),
            function_annotation=Enzyme.Const,
        ),
    )
    push!(
        ADTYPES,
        AutoEnzyme(;
            mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation=Enzyme.Const,
        ),
    )
end

# Check that ADTypeCheckContext itself works as expected.
@testset "ADTypeCheckContext" begin
    @model test_model() = x ~ Normal(0, 1)
    tm = test_model()
    adtypes = (
        AutoForwardDiff(),
        AutoReverseDiff(),
        # Don't need to test Mooncake as it doesn't use tracer types
    )
    for actual_adtype in adtypes
        sampler = HMC(0.1, 5; adtype=actual_adtype)
        for expected_adtype in adtypes
            contextualised_tm = DynamicPPL.contextualize(
                tm, ADTypeCheckContext(expected_adtype, tm.context)
            )
            @testset "Expected: $expected_adtype, Actual: $actual_adtype" begin
                if actual_adtype == expected_adtype
                    # Check that this does not throw an error.
                    sample(contextualised_tm, sampler, 2)
                else
                    @test_throws AbstractWrongADBackendError sample(
                        contextualised_tm, sampler, 2
                    )
                end
            end
        end
    end
end

@testset verbose = true "AD / ADTypeCheckContext" begin
    # This testset ensures that samplers or optimisers don't accidentally
    # override the AD backend set in it.
    @testset "adtype=$adtype" for adtype in ADTYPES
        seed = 123
        alg = HMC(0.1, 10; adtype=adtype)
        m = DynamicPPL.contextualize(
            gdemo_default, ADTypeCheckContext(adtype, gdemo_default.context)
        )
        # These will error if the adbackend being used is not the one set.
        sample(StableRNG(seed), m, alg, 10)
        maximum_likelihood(m; adtype=adtype)
        maximum_a_posteriori(m; adtype=adtype)
    end
end

@testset verbose = true "AD / GibbsContext" begin
    # Gibbs sampling needs some extra AD testing because the models are
    # executed with GibbsContext and a subsetted varinfo. (see e.g.
    # `gibbs_initialstep_recursive` and `gibbs_step_recursive` in
    # src/mcmc/gibbs.jl -- the code here mimics what happens in those
    # functions)
    @testset "adtype=$adtype" for adtype in ADTYPES
        @testset "model=$(model.f)" for model in DEMO_MODELS
            # All the demo models have variables `s` and `m`, so we'll pretend
            # that we're using a Gibbs sampler where both of them are sampled
            # with a gradient-based sampler (say HMC(0.1, 10)).
            # This means we need to construct one with only `s`, and one model with
            # only `m`.
            global_vi = DynamicPPL.VarInfo(model)
            @testset for varnames in ([@varname(s)], [@varname(m)])
                @info "Testing Gibbs AD with model=$(model.f), varnames=$varnames"
                conditioned_model = Turing.Inference.make_conditional(
                    model, varnames, deepcopy(global_vi)
                )
                rng = StableRNG(123)
                @test run_ad(model, adtype; test=true, benchmark=false) isa Any
            end
        end
    end
end

@testset verbose = true "AD / Gibbs sampling" begin
    # Make sure that Gibbs sampling doesn't fall over when using AD.
    @testset "adtype=$adtype" for adtype in ADTYPES
        spl = Gibbs(
            @varname(s) => HMC(0.1, 10; adtype=adtype),
            @varname(m) => HMC(0.1, 10; adtype=adtype),
        )
        @testset "model=$(model.f)" for model in DEMO_MODELS
            @test sample(model, spl, 2) isa Any
        end
    end
end

end # module
