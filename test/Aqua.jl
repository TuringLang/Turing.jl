module AquaTests

using Aqua: Aqua
using Turing

# TODO(mhauru) We skip testing for method ambiguities because it catches a lot of problems
# in dependencies. Would like to check it for just Turing.jl itself though.
Aqua.test_all(Turing; ambiguities=false)

end
