###########
# VarName #
###########
"""
```
struct VarName{sym}
    csym      ::    Symbol
    indexing  ::    String
    counter   ::    Int
end
```

A variable identifier. Every variable has a symbol `sym`, indices `indexing`, and 
internal fields: `csym` and `counter`. The Julia variable in the model corresponding to 
`sym` can refer to a single value or to a hierarchical array structure of univariate, 
multivariate or matrix variables. `indexing` stores the indices that can access the 
random variable from the Julia variable. 

Examples:

- `x[1] ~ Normal()` will generate a `VarName` with `sym == :x` and `indexing == "[1]"`.
- `x[:,1] ~ MvNormal(zeros(2))` will generate a `VarName` with `sym == :x` and 
 `indexing == "[Colon(), 1]"`.
- `x[:,1][2] ~ Normal()` will generate a `VarName` with `sym == :x` and 
 `indexing == "[Colon(), 1][2]"`.
"""
struct VarName{sym}
    csym      ::    Symbol        # symbol generated in compilation time
    indexing  ::    String        # indexing
    counter   ::    Int           # counter of same {csym, uid}
end

abstract type AbstractVarInfo end

####################
# VarInfo metadata #
####################

"""
The `Metadata` struct stores some metadata about the parameters of the model. This helps 
query certain information about a variable, such as its distribution, which samplers 
sample this variable, its value and whether this value is transformed to real space or 
not.

Let `md` be an instance of `Metadata`:
- `md.vns` is the vector of all `VarName` instances.
- `md.idcs` is the dictionary that maps each `VarName` instance to its index in 
 `md.vns`, `md.ranges` `md.dists`, `md.orders` and `md.flags`.
- `md.vns[md.idcs[vn]] == vn`.
- `md.dists[md.idcs[vn]]` is the distribution of `vn`.
- `md.gids[md.idcs[vn]]` is the set of algorithms used to sample `vn`. This is used in 
 the Gibbs sampling process.
- `md.orders[md.idcs[vn]]` is the number of `observe` statements before `vn` is sampled.
- `md.ranges[md.idcs[vn]]` is the index range of `vn` in `md.vals`.
- `md.vals[md.ranges[md.idcs[vn]]]` is the vector of values of corresponding to `vn`.
- `md.flags` is a dictionary of true/false flags. `md.flags[flag][md.idcs[vn]]` is the 
 value of `flag` corresponding to `vn`. 

To make `md::Metadata` type stable, all the `md.vns` must have the same symbol 
and distribution type. However, one can have a Julia variable, say `x`, that is a 
matrix or a hierarchical array sampled in partitions, e.g. 
`x[1][:] ~ MvNormal(zeros(2), 1.0); x[2][:] ~ MvNormal(ones(2), 1.0)`, and is managed by 
a single `md::Metadata` so long as all the distributions on the RHS of `~` are of the 
same type. Type unstable `Metadata` will still work but will have inferior performance. 
When sampling, the first iteration uses a type unstable `Metadata` for all the 
variables then a specialized `Metadata` is used for each symbol along with a function 
barrier to make the rest of the sampling type stable.
"""
struct Metadata{TIdcs <: Dict{<:VarName,Int}, TDists <: AbstractVector{<:Distribution}, TVN <: AbstractVector{<:VarName}, TVal <: AbstractVector{<:Real}, TGIds <: AbstractVector{Set{Selector}}}
    # Mapping from the `VarName` to its integer index in `vns`, `ranges` and `dists`
    idcs        ::    TIdcs # Dict{<:VarName,Int}

    # Vector of identifiers for the random variables, where `vns[idcs[vn]] == vn`
    vns         ::    TVN # AbstractVector{<:VarName}

    # Vector of index ranges in `vals` corresponding to `vns`
    # Each `VarName` `vn` has a single index or a set of contiguous indices in `vals`
    ranges      ::    Vector{UnitRange{Int}}

    # Vector of values of all the univariate, multivariate and matrix variables
    # The value(s) of `vn` is/are `vals[ranges[idcs[vn]]]`
    vals        ::    TVal # AbstractVector{<:Real}

    # Vector of distributions correpsonding to `vns`
    dists       ::    TDists # AbstractVector{<:Distribution}

    # Vector of sampler ids corresponding to `vns`
    # Each random variable can be sampled using multiple samplers, e.g. in Gibbs, hence the `Set`
    gids        ::    TGIds # AbstractVector{Set{Selector}}

    # Number of `observe` statements before each random variable is sampled
    orders      ::    Vector{Int}

    # Each `flag` has a `BitVector` `flags[flag]`, where `flags[flag][i]` is the true/false flag value corresonding to `vns[i]`
    flags       ::    Dict{String, BitVector}
end

###########
# VarInfo #
###########

"""
```
struct VarInfo{Tmeta, Tlogp} <: AbstractVarInfo
    metadata::Tmeta
    logp::Base.RefValue{Tlogp}
    num_produce::Base.RefValue{Int}
end
```

A light wrapper over one or more instances of `Metadata`. Let `vi` be an instance of 
`VarInfo`. If `vi isa VarInfo{<:Metadata}`, then only one `Metadata` instance is used 
for all the sybmols. `VarInfo{<:Metadata}` is aliased `UntypedVarInfo`. If 
`vi isa VarInfo{<:NamedTuple}`, then `vi.metadata` is a `NamedTuple` that maps each 
symbol used on the LHS of `~` in the model to its `Metadata` instance. The latter allows 
for the type specialization of `vi` after the first sampling iteration when all the 
symbols have been observed. `VarInfo{<:NamedTuple}` is aliased `TypedVarInfo`.

Note: It is the user's responsibility to ensure that each "symbol" is visited at least 
once whenever the model is called, regardless of any stochastic branching. Each symbol 
refers to a Julia variable and can be a hierarchical array of many random variables, e.g. `x[1] ~ ...` and `x[2] ~ ...` both have the same symbol `x`.
"""
struct VarInfo{Tmeta, Tlogp} <: AbstractVarInfo
    metadata::Tmeta
    logp::Base.RefValue{Tlogp}
    num_produce::Base.RefValue{Int}
end
const UntypedVarInfo = VarInfo{<:Metadata}
const TypedVarInfo = VarInfo{<:NamedTuple}
