GKernel(var) = (x) -> Normal(x, sqrt.(var))

function randr(vi::Turing.VarInfo,
               vn::Turing.VarName,
               dist::Distribution,
               spl::Turing.Sampler,
               count::Bool = false)
    if ~haskey(vi, vn)
        r = rand(dist)
        Turing.push!(vi, vn, r, dist, spl)
        return r
    elseif is_flagged(vi, vn, "del")
        unset_flag!(vi, vn, "del")
        r = rand(dist)
        Turing.RandomVariables.setval!(vi, Turing.vectorize(dist, r), vn)
        return r
    else
        if count Turing.checkindex(vn, vi, spl) end
        Turing.updategid!(vi, vn, spl)
        return vi[vn]
    end
end

function insdelim(c, deli=",")
	return reduce((e, res) -> append!(e, [res, deli]), c; init = [])[1:end-1]
end
