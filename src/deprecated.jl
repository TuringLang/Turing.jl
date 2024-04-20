export setadbackend, setchunksize, setadsafe

function setadbackend(::Union{Symbol, Val})
    Base.depwarn("`ADBACKEND` and `setbackend` are deprecated. Please specify the chunk size directly in the sampler constructor, e.g., `HMC(0.1, 5; adtype=AutoForwardDiff())`.\n This function has no effects.", :setbackend; force=true)
    nothing
end

function setchunksize(::Int)
    Base.depwarn("`CHUNKSIZE` and `setchunksize` are deprecated. Please specify the chunk size directly in the sampler constructor, e.g., `HMC(0.1, 5; adtype=AutoForwardDiff())`.\n This function has no effects.", :setchunksize; force=true)
    nothing
end

function setrdcache(::Union{Bool, Val})
    Base.depwarn("`RDCACHE` and `setrdcache` are deprecated. Please specify if you wish to use compiled tape for ReverseDiff directly in the sampler constructor, e.g., `HMC(0.1, 5; adtype=AutoReverseDiff(false))`.\n This function has no effects.", :setrdcache; force=true)
    nothing
end

function setadsafe(::Bool)
    Base.depwarn("`ADSAFE` and `setadsafe` are outdated and no longer in use.", :setadsafe; force=true)
    nothing
end

Base.@deprecate_binding Core Essential false
