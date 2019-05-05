module RandomVariables

using ...Turing: Turing, CACHERESET, CACHEIDCS, CACHERANGES, Model,
    AbstractSampler, Sampler, SampleFromPrior,
    Selector
using ...Utilities: vectorize, reconstruct, reconstruct!
using Bijectors: SimplexDistribution, link, invlink
using Distributions

import Base:    string, 
                Symbol, 
                ==, 
                hash, 
                in, 
                getindex, 
                setindex!, 
                push!, 
                show, 
                isempty, 
                empty!, 
                getproperty, 
                setproperty!, 
                keys, 
                haskey

export  VarName, 
        AbstractVarInfo,
        VarInfo,
        UntypedVarInfo,
        getlogp, 
        setlogp!, 
        set_retained_vns_del_by_spl!, 
        resetlogp!, 
        is_flagged, 
        unset_flag!, 
        setgid!, 
        setorder!, 
        updategid!, 
        acclogp!, 
        istrans, 
        link!, 
        invlink!

include("types.jl")
include("internal.jl")
include("api.jl")

end
