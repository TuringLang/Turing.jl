Sampler
=========

.. function:: IS(n_particles::Int)

   Importance sampler.

   Usage:

   .. code-block:: julia

       IS(1000)

   Example:

   .. code-block:: julia

       @model example begin
         ...
       end

       sample(example, IS(1000))

.. function:: SMC(n_particles::Int)

   Sequential Monte Carlo sampler.

   Usage:

   .. code-block:: julia

       SMC(1000)

   Example:

   .. code-block:: julia

       @model example begin
         ...
       end

       sample(example, SMC(1000))

.. function:: PG(n_particles::Int, n_iterations::Int)

   Particle Gibbs sampler.

   Usage:

   .. code-block:: julia

       PG(100, 100)

   Example:

   .. code-block:: julia

       @model example begin
         ...
       end

       sample(example, PG(100, 100))

.. function:: HMC(n_samples::Int64, lf_size::Float64, lf_num::Int64)

   Hamiltonian Monte Carlo sampler.

   Usage:

   .. code-block:: julia

       HMC(1000, 0.05, 10)

   Example:

   .. code-block:: julia

       @model example begin
         ...
       end

       sample(example, HMC(1000, 0.05, 10))

