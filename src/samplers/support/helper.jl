##########
# Others #
##########

# VarInfo to Sample
@inline Sample(vi::VarInfo) = begin
  value = Dict{Symbol, Any}() # value is named here because of Sample has a field called value
  for vn in keys(vi)
    r = getuval(vi, vn)
    value[sym(vn)] = realpart(r == nothing? vi[vn] : r)
  end

  # NOTE: do we need to check if lp is 0?
  value[:lp] = realpart(getlogp(vi))



  if ~isempty(vi.pred)
    for sym in keys(vi.pred)
      # if ~haskey(sample.value, sym)
        value[sym] = vi.pred[sym]
      # end
    end
    # TODO: check why 1. 2. cause errors
    # TODO: which one is faster?
    # 1. Using empty!
    # empty!(vi.pred)
    # 2. Reassign an enmtpy dict
    # vi.pred = Dict{Symbol,Any}()
    # 3. Do nothing?
  end

  Sample(0.0, value)
end

# VarInfo, combined with spl.info, to Sample
@inline Sample(vi::VarInfo, spl::Sampler) = begin
  s = Sample(vi)

  if haskey(spl.info, :ϵ)
    s.value[:epsilon] = spl.info[:ϵ][end]
  end

  if haskey(spl.info, :lf_num)
    s.value[:lf_num] = spl.info[:lf_num][end]
  end

  s
end
