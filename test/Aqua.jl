module AquaTests

using Aqua: Aqua
using Libtask: Libtask
using Turing

# We test ambiguities specifically only for Turing, because testing ambiguities for all
# packages in the environment leads to a lot of ambiguities from dependencies that we cannot
# control.
#
# `Libtask.might_produce` is excluded because the `@might_produce` macro generates a lot of
# ambiguities that will never happen in practice.
#
# Specifically, when you write `@might_produce f` for a function `f` that has methods that
# take keyword arguments, we have to generate a `might_produce` method for
# `Type{<:Tuple{<:Function,...,typeof(f)}}`. There is no way to circumvent this: see
# https://github.com/TuringLang/Libtask.jl/issues/197. This in turn will cause method
# ambiguities with any other function, say `g`, for which
# `::Type{<:Tuple{typeof(g),Vararg}}` is marked as produceable.
#
# To avoid the method ambiguities, we *could* manually spell out `might_produce` methods for
# each method of `g` manually instead of using Vararg, but that would be both very verbose
# and fragile. It would also not provide any real benefit since those ambiguities are not
# meaningful in practice (in particular, to trigger this we would need to call `g(..., f)`,
# which is incredibly unlikely).
Aqua.test_ambiguities([Turing]; exclude=[Libtask.might_produce])
Aqua.test_all(Turing; ambiguities=false)

end
