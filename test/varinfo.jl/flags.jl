using Turing: is_flagged, set_flag!, unset_flag!
using Turing: VarInfo, VarName
using Distributions
using Base.Test


vi = VarInfo()
vn_x = VarName(gensym(), :x, "", 1)
dist = Normal(0, 1)
r = rand(dist)
gid = 0

push!(vi, vn_x, r, dist, gid)

# del is set by default
@test is_flagged(vi, vn_x, "del") == false

set_flag!(vi, vn_x, "del")
@test is_flagged(vi, vn_x, "del") == true

unset_flag!(vi, vn_x, "del")
@test is_flagged(vi, vn_x, "del") == false