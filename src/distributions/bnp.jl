immutable CRP
  counts::TArray{Float64, 1}
  alpha::Float64
end

CRP(alpha) = CRP(convert(TArray, [alpha]),  alpha)

function randclass(urn::CRP)
  counts = localcopy(urn.counts)
  weights = counts ./ sum(counts)
  @assume c ~ Categorical(weights)
  if c == length(counts)
    urn.counts[end] = 1
    push!(urn.counts, urn.alpha)
  elseif c < length(counts) && c > 0
    urn.counts[c] += 1
  else
    #println(weights)
    #println("before assume: $(previous_db)")
    #println("after assume: $(current_randomdb())")
    error("RANDCLASS: class id: $c, urn.counts: $(urn.counts), randomdb: $(current_randomdb())")
  end
  return Int64(c)::Int64
end

Base.deepcopy(urn::CRP) = CRP(deepcopy(urn.counts), urn.alpha)

export CRP, randclass
