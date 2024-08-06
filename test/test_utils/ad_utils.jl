module ADUtils

using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Test: Test
using Tracker: Tracker
using Turing: Turing
using Turing: DynamicPPL
using Zygote: Zygote

export ADTypeCheckContext

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
    # TODO(mhauru) Zygote.Dual is actually the same as ForwardDiff.Dual, so can't
    # distinguish between the two.
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
    IncompatibleADTypeError

An error thrown when an element type is encountered that is unexpected for the given ADType.
"""
struct IncompatibleADTypeError <: Exception
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
        if !any(adtype .<: keys(eltypes_by_adtype))
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
    value, logp, vi = DynamicPPL.tilde_assume(
        DynamicPPL.childcontext(context), right, vn, vi
    )
    check_adtype(context, vi)
    return value, logp, vi
end

function DynamicPPL.tilde_assume(rng, context::ADTypeCheckContext, sampler, right, vn, vi)
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
    rng, context::ADTypeCheckContext, sampler, right, left, vn, vi
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
    contextualised_tm = DynamicPPL.contextualize(
        tm, ADTypeCheckContext(Turing.AutoForwardDiff(), tm.context)
    )
    # This should not throw an error since we are using ForwardDiff as expected.
    Turing.sample(contextualised_tm, Turing.NUTS(; adtype=Turing.AutoForwardDiff()), 100)
    # Using ReverseDiff when ForwardDiff is expected should throw an error.
    Test.@test_throws IncompatibleADTypeError Turing.sample(
        contextualised_tm, Turing.NUTS(; adtype=Turing.AutoReverseDiff()), 100
    )
end

end
