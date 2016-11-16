Breif Introduction to Samplers
============

Sequential Monte Carlo
----------------------

Sequential Monte Carlo (SMC) is a set of simulation-based inference algorithms allowing us to approximate any distributions sequentially. The aim of SMC is to estimate unknown quantities from some observations, where the unobserved signal is modelled by a Markov process and the observation is assumed to be conditionally independent given the unobserved signal [1].

Particle Gibbs
--------------

The Particle Gibbs (PG) method is a SMC based method that runs multiple passes of the SMC algorithm, where each pass is conditional on the trajectory sampled at the last run of the SMC sampler [2].

Hamiltonian Monte Carlo
-----------------------

Hamiltonian Monte Carlo (HMC) is a Markov Chain Monte Carlo (MCMC) method for generating samples from a probability distribution for which direct sampling is difficult. It was originally devised by Simon Duane, A.D. Kennedy, Brian Pendleton and Duncan Roweth in 1987 [3].

Reference
---------

[1] Arnaud Doucet, Nando De Freitas, and NJ Gordon. An introduction to sequential monte carlo method. SMC in Practice, 2001.
[2] Darren Wilkinson. Introduction to the particle gibbs sampler. `<https://darrenjw.wordpress.com/2014/01/25/>introduction-to-the-particle-gibbs-sampler/`_ , 2014. [Online; accessed 10-July-2016]
[3] MacKay, David JC. Information theory, inference and learning algorithms. Cambridge university press, 2003.
