doc"""
    get_gradient_dict(spl :: GradientSampler)

Function to generate the gradient dictionary, with each prior map to its derivative of the logjoint. This function uses chunk-wise forward AD with a chunk of size 10, which is limited by the ForwardDiff package.

Example:

```julia
function Base.run(spl :: Sampler{HMC})
  ...
  val∇E = get_gradient_dict(spl.priors, spl.model)
  ...
end
```
"""
function get_gradient_dict(varInfo::VarInfo, model::Function, data=Dict(), spl=nothing)
  # Initialisation
  val∇E = Dict{Var, Vector{Float64}}()
  # Split keys(values) into CHUNKSIZE, CHUNKSIZE, CHUNKSIZE, m-size chunks,
  dprintln(4, "making chunks...")
  prior_key_chunks = []
  key_chunk = []
  prior_dim = 0
  for k in keys(varInfo)
    if spl == nothing || isempty(spl.alg.space) || k.sym in spl.alg.space
      l = length(varInfo[k])
      if prior_dim + l > CHUNKSIZE
        # Store the old chunk
        push!(prior_key_chunks, (key_chunk, prior_dim))
        # Initialise new chunk
        key_chunk = []
        prior_dim = 0
        # Update
        push!(key_chunk, k)
        prior_dim += l
      else
        # Update
        push!(key_chunk, k)
        prior_dim += l
      end
    end
  end
  if length(key_chunk) != 0
    push!(prior_key_chunks, (key_chunk, prior_dim))  # push the last chunk
  end
  # chunk-wise forward AD
  for (key_chunk, prior_dim) in prior_key_chunks
    # Set dual part correspondingly
    dprintln(4, "set dual...")
    dps = eye(prior_dim)      # dualpart values to set
    prior_count = 1
    for k in keys(varInfo)
      l = length(varInfo[k])
      reals = realpart(varInfo[k])
      val_vect = varInfo[k]   # get a reference for the value vector
      if k in key_chunk       # to graidnet variables
        dprintln(5, "making dual...")
        for i = 1:l
          val_vect[i] = Dual(reals[i], dps[1:end, prior_count]...)
          prior_count += 1    # count
        end
        dprintln(5, "make dual done")
      else                    # other varilables (not for gradient info)
        for i = 1:l
          val_vect[i] = Dual{prior_dim, Float64}(reals[i])
        end
      end
    end
    # Run the model
    dprintln(4, "run model...")
    varInfo = model(data, varInfo, spl)
    # Collect gradient
    dprintln(4, "collect dual...")
    prior_count = 1
    for k in key_chunk
      dprintln(5, "for each prior...")
      l = length(varInfo[k])
      reals = realpart(varInfo[k])
      # To store the gradient vector
      g = zeros(l)
      for i = 1:l
        # Collect
        dprintln(5, "taking from logjoint...")
        g[i] = dualpart(-varInfo.logjoint)[prior_count]
        # Count
        prior_count += 1
      end
      val∇E[k] = g
    end
    # Reset logjoint
    varInfo.logjoint = Dual(0)
  end
  # Return
  return val∇E
end
