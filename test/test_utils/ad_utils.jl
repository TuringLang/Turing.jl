module ADUtils

import Pkg
import Turing

"""
All the ADTypes on which we want to run the tests.
"""
adbackends = [
    Turing.AutoForwardDiff(; chunksize=0), Turing.AutoReverseDiff(; compile=false)
]

# Tapir isn't supported for older Julia versions, hence the check.
install_tapir = isdefined(Turing.AutoTapir)
if install_tapir
    # TODO(mhauru) Is there a better way to install optional dependencies like this?
    Pkg.add("Tapir")
    push!(adbackends, Turing.AutoTapir())
end

end
