module VarReplay

import Base: string, isequal, ==, hash, getindex, setindex!, push!, show, isempty

using ...Turing: CACHERESET, CACHEIDCS, CACHERANGES
using ...Samplers
using Distributions
using Parameters: @unpack
using ...Utilities
using Bijectors

export  VarName, 
        VarInfo, 
        UntypedVarInfo, 
        TypedVarInfo, 
        AbstractVarInfo, 
        uid, 
        sym, 
        getlogp, 
        set_retained_vns_del_by_spl!, 
        resetlogp!, 
        is_flagged, 
        unset_flag!, 
        setgid!, 
        copybyindex, 
        setorder!, 
        updategid!, 
        acclogp!, 
        istrans, 
        link!, 
        invlink!, 
        setlogp!, 
        getranges, 
        getrange, 
        getvns, 
        getval,
        NewVarInfo

include("varinfo.jl")
include("typed_varinfo.jl")

end # module
