doc"""
    get_gradient_dict(spl :: Sampler)

Function to generate the gradient dictionary, with each prior map to its derivative of the logjoint. This function uses chunk-wise forward AD with a chunk of size 10, which is limited by the ForwardDiff package.

Example:

```julia
function Base.run(spl :: Sampler{HMC})
  ...
  val∇E = get_gradient_dict(spl)
  ...
end
```
"""
function get_gradient_dict(spl :: Sampler)
  # Initialisation
  val∇E = Dict{Any, Any}()
  # Split keys(spl.priors) into 10, 10, 10, m-size chunks
  dprintln(5, "making chunks...")
  prior_key_chunks = []
  key_chunk = []
  prior_dim = 0
  for k in keys(spl.priors)
    l = length(spl.priors[k])
    if prior_dim + l > 10
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
  if length(key_chunk) != 0
    push!(prior_key_chunks, (key_chunk, prior_dim))  # push the last chunk
  end
  # chunk-wise forward AD
  for (key_chunk, prior_dim) in prior_key_chunks
    # Set dual part correspondingly
    dprintln(5, "set dual...")
    prior_count = 1
    for k in keys(spl.priors)
      l = length(spl.priors[k])
      reals = realpart(spl.priors[k])
      val_vect = spl.priors[k]   # get the value vector

      if k in key_chunk   # to graidnet variables
        for i = 1:l
          dprintln(5, "making dual...")
          val_vect[i] = make_dual(prior_dim, reals[i], prior_count)
          dprintln(5, "make dual done")
          # Count
          prior_count += 1
        end
      else                # other varilables
        for i = 1:l
          val_vect[i] = Dual{prior_dim, Float64}(reals[i])
        end
      end
    end
    # Run the model
    dprintln(5, "run model...")
    consume(Task(spl.model))
    # Collect gradient
    dprintln(5, "collect dual...")
    prior_count = 1
    for k in key_chunk
      l = length(spl.priors[k])
      reals = realpart(spl.priors[k])
      # To store the gradient vector
      g = zeros(l)
      for i = 1:l
        # Collect
        g[i] = dualpart(-spl.logjoint)[prior_count]
        # Count
        prior_count += 1
      end
      val∇E[k] = g
    end
    # Reset logjoint
    spl.logjoint = Dual(0)
  end
  # Return
  return val∇E
end
