module ADUtils

import Turing

"""
All the ADTypes on which we want to run the tests.
"""
adbackends = [
    Turing.AutoForwardDiff(; chunksize=0), Turing.AutoReverseDiff(; compile=false)
]
# AutoTapir isn't supported for older Julia versions, hence the check.
if isdefined(Turing.AutoTapir)
    push!(backends, Turing.AutoTapir())
end

end
