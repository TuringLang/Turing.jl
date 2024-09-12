module ADUtils

using ForwardDiff: ForwardDiff
using Pkg: Pkg
using Random: Random
using ReverseDiff: ReverseDiff
using Test: Test
using Tracker: Tracker
using Turing: Turing
using Turing: DynamicPPL
using Zygote: Zygote

export ADTypeCheckContext, adbackends

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Stuff for checking that the right AD backend is being used.

"""Element types that are always valid for a VarInfo regardless of ADType."""
const always_valid_eltypes = (AbstractFloat, AbstractIrrational, Integer, Rational)

"""A dictionary mapping ADTypes to the element types they use."""
const eltypes_by_adtype = Dict(
    Turing.AutoForwardDiff => (ForwardDiff.Dual,),
    Turing.AutoReverseDiff => (
        ReverseDiff.TrackedArray,
        ReverseDiff.TrackedMatrix,
        ReverseDiff.TrackedReal,
        ReverseDiff.TrackedStyle,
        ReverseDiff.TrackedType,
        ReverseDiff.TrackedVecOrMat,
        ReverseDiff.TrackedVector,
    ),
    # Zygote.Dual is actually the same as ForwardDiff.Dual, so can't distinguish between the
    # two by element type. However, we have other checks for Zygote, see check_adtype.
    Turing.AutoZygote => (Zygote.Dual,),
    Turing.AutoTracker => (
        Tracker.Tracked,
        Tracker.TrackedArray,
        Tracker.TrackedMatrix,
        Tracker.TrackedReal,
        Tracker.TrackedStyle,
        Tracker.TrackedVecOrMat,
        Tracker.TrackedVector,
    ),
)

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

As a current short-coming, this context can not distinguish between ForwardDiff and Zygote.
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

When Zygote is being used, we also more explicitly check that `adtype(context)` is
`AutoZygote`. This is because Zygote uses the same element type as ForwardDiff, so we can't
discriminate between the two based on element type alone. This function will still fail to
catch cases where Zygote is supposed to be used, but ForwardDiff is used instead.

Throw an `IncompatibleADTypeError` if an incompatible element type is encountered, or
`WrongADBackendError` if Zygote is used unexpectedly.
"""
function check_adtype(context::ADTypeCheckContext, vi::DynamicPPL.AbstractVarInfo)
    Zygote.hook(vi) do _
        if !(adtype(context) <: Turing.AutoZygote)
            throw(WrongADBackendError(Turing.AutoZygote, adtype(context)))
        end
    end

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
    value, logp, vi = DynamicPPL.tilde_assume(
        DynamicPPL.childcontext(context), right, vn, vi
    )
    check_adtype(context, vi)
    return value, logp, vi
end

function DynamicPPL.tilde_assume(
    rng::Random.AbstractRNG, context::ADTypeCheckContext, sampler, right, vn, vi
)
    value, logp, vi = DynamicPPL.tilde_assume(
        rng, DynamicPPL.childcontext(context), sampler, right, vn, vi
    )
    check_adtype(context, vi)
    return value, logp, vi
end

function DynamicPPL.tilde_observe(context::ADTypeCheckContext, right, left, vi)
    logp, vi = DynamicPPL.tilde_observe(DynamicPPL.childcontext(context), right, left, vi)
    check_adtype(context, vi)
    return logp, vi
end

function DynamicPPL.tilde_observe(context::ADTypeCheckContext, sampler, right, left, vi)
    logp, vi = DynamicPPL.tilde_observe(
        DynamicPPL.childcontext(context), sampler, right, left, vi
    )
    check_adtype(context, vi)
    return logp, vi
end

function DynamicPPL.dot_tilde_assume(context::ADTypeCheckContext, right, left, vn, vi)
    value, logp, vi = DynamicPPL.dot_tilde_assume(
        DynamicPPL.childcontext(context), right, left, vn, vi
    )
    check_adtype(context, vi)
    return value, logp, vi
end

function DynamicPPL.dot_tilde_assume(
    rng::Random.AbstractRNG, context::ADTypeCheckContext, sampler, right, left, vn, vi
)
    value, logp, vi = DynamicPPL.dot_tilde_assume(
        rng, DynamicPPL.childcontext(context), sampler, right, left, vn, vi
    )
    check_adtype(context, vi)
    return value, logp, vi
end

function DynamicPPL.dot_tilde_observe(context::ADTypeCheckContext, right, left, vi)
    logp, vi = DynamicPPL.dot_tilde_observe(
        DynamicPPL.childcontext(context), right, left, vi
    )
    check_adtype(context, vi)
    return logp, vi
end

function DynamicPPL.dot_tilde_observe(context::ADTypeCheckContext, sampler, right, left, vi)
    logp, vi = DynamicPPL.dot_tilde_observe(
        DynamicPPL.childcontext(context), sampler, right, left, vi
    )
    check_adtype(context, vi)
    return logp, vi
end

# Check that the ADTypeCheckContext works as expected.
Test.@testset "ADTypeCheckContext" begin
    Turing.@model test_model() = x ~ Turing.Normal(0, 1)
    tm = test_model()
    adtypes = (
        Turing.AutoForwardDiff(),
        Turing.AutoReverseDiff(),
        Turing.AutoZygote(),
        Turing.AutoTracker(),
    )
    for actual_adtype in adtypes
        sampler = Turing.HMC(0.1, 5; adtype=actual_adtype)
        for expected_adtype in adtypes
            if (
                actual_adtype == Turing.AutoForwardDiff() &&
                expected_adtype == Turing.AutoZygote()
            )
                # TODO(mhauru) We are currently unable to check this case.
                continue
            end
            contextualised_tm = DynamicPPL.contextualize(
                tm, ADTypeCheckContext(expected_adtype, tm.context)
            )
            Test.@testset "Expected: $expected_adtype, Actual: $actual_adtype" begin
                if actual_adtype == expected_adtype
                    # Check that this does not throw an error.
                    Turing.sample(contextualised_tm, sampler, 2)
                else
                    Test.@test_throws AbstractWrongADBackendError Turing.sample(
                        contextualised_tm, sampler, 2
                    )
                end
            end
        end
    end
end

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# List of AD backends to test.

"""
All the ADTypes on which we want to run the tests.
"""
adbackends = [
    Turing.AutoForwardDiff(; chunksize=0), Turing.AutoReverseDiff(; compile=false)
]

# Tapir isn't supported for older Julia versions, hence the check.
install_tapir = isdefined(Turing, :AutoTapir)
if install_tapir
    # TODO(mhauru) Is there a better way to install optional dependencies like this?
    Pkg.add(; name="Tapir", version="0.2.48")
    using Tapir
    push!(adbackends, Turing.AutoTapir(false))
    push!(eltypes_by_adtype, Turing.AutoTapir => (Tapir.CoDual,))
end

end
