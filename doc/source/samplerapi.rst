Sampler
=========

.. function:: IS(n_particles::Int)

   Importance sampling algorithm object.

   * ``n_particles`` is the number of particles to use

   Usage:

   .. code-block:: julia

       IS(1000)

   Example:

   .. code-block:: julia

       # Define a simple Normal model with unknown mean and variance.
       @model gdemo(x) = begin
         s ~ InverseGamma(2,3)
         m ~ Normal(0,sqrt(s))
         x[1] ~ Normal(m, sqrt(s))
         x[2] ~ Normal(m, sqrt(s))
         return s, m
       end

       sample(gdemo([1.5, 2]), IS(1000))

.. function:: SMC(n_particles::Int)

   Sequential Monte Carlo sampler.

   Usage:

   .. code-block:: julia

       SMC(1000)

   Example:

   .. code-block:: julia

       # Define a simple Normal model with unknown mean and variance.
       @model gdemo(x) = begin
         s ~ InverseGamma(2,3)
         m ~ Normal(0,sqrt(s))
         x[1] ~ Normal(m, sqrt(s))
         x[2] ~ Normal(m, sqrt(s))
         return s, m
       end

       sample(gdemo([1.5, 2]), SMC(1000))

.. function:: PG(n_particles::Int, n_iters::Int)

   Particle Gibbs sampler.

   Usage:

   .. code-block:: julia

       PG(100, 100)

   Example:

   .. code-block:: julia

       # Define a simple Normal model with unknown mean and variance.
       @model gdemo(x) = begin
         s ~ InverseGamma(2,3)
         m ~ Normal(0,sqrt(s))
         x[1] ~ Normal(m, sqrt(s))
         x[2] ~ Normal(m, sqrt(s))
         return s, m
       end

       sample(gdemo([1.5, 2]), PG(100, 100))

.. function:: HMC(n_iters::Int, epsilon::Float64, tau::Int)

   Hamiltonian Monte Carlo sampler.

   Usage:

   .. code-block:: julia

       HMC(1000, 0.05, 10)

   Example:

   .. code-block:: julia

       # Define a simple Normal model with unknown mean and variance.
       @model gdemo(x) = begin
         s ~ InverseGamma(2,3)
         m ~ Normal(0,sqrt(s))
         x[1] ~ Normal(m, sqrt(s))
         x[2] ~ Normal(m, sqrt(s))
         return s, m
       end

       sample(gdemo([1.5, 2]), HMC(1000, 0.05, 10))

.. function:: HMCDA(n_iters::Int, n_adapt::Int, delta::Float64, lambda::Float64)

   Hamiltonian Monte Carlo sampler wiht Dual Averaging algorithm.

   Usage:

   .. code-block:: julia

       HMCDA(1000, 200, 0.65, 0.3)

   Example:

   .. code-block:: julia

       # Define a simple Normal model with unknown mean and variance.
       @model gdemo(x) = begin
         s ~ InverseGamma(2,3)
         m ~ Normal(0,sqrt(s))
         x[1] ~ Normal(m, sqrt(s))
         x[2] ~ Normal(m, sqrt(s))
         return s, m
       end

       sample(gdemo([1.5, 2]), HMCDA(1000, 200, 0.65, 0.3))

.. function:: NUTS(n_iters::Int, n_adapt::Int, delta::Float64)

   No-U-Turn Sampler (NUTS) sampler.

   Usage:

   .. code-block:: julia

       NUTS(1000, 200, 0.65)

   Example:

   .. code-block:: julia

       # Define a simple Normal model with unknown mean and variance.
       @model gdemo(x) = begin
         s ~ InverseGamma(2,3)
         m ~ Normal(0,sqrt(s))
         x[1] ~ Normal(m, sqrt(s))
         x[2] ~ Normal(m, sqrt(s))
         return s, m
       end

       sample(gdemo([1.5, 2]), NUTS(1000, 200, 0.65))

.. function:: Gibbs(n_iters, alg_1, alg_2)

   Compositional MCMC interface.

   Usage:

   .. code-block:: julia

       alg = Gibbs(1000, HMC(1, 0.2, 3, :v1), PG(20, 1, :v2))

