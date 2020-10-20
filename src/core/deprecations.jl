function setadbackend(::Val{:forward_diff})
    Base.depwarn("`Turing.setadbackend(:forward_diff)` is deprecated. Please use `Turing.setadbackend(:forwarddiff)` to use `ForwardDiff`.", :setadbackend)
    setadbackend(Val(:forwarddiff))
end

function setadbackend(::Val{:reverse_diff})
    Base.depwarn("`Turing.setadbackend(:reverse_diff)` is deprecated. Please use `Turing.setadbackend(:tracker)` to use `Tracker` or `Turing.setadbackend(:reversediff)` to use `ReverseDiff`. To use `ReverseDiff`, please make sure it is loaded separately with `using ReverseDiff`.",  :setadbackend)
    setadbackend(Val(:tracker))
end
