using Test, Turing

# Run HMC with chunk_size=1
chain = sample(gdemo_default, HMC(300, 0.1, 1))

# Runs
sample(gdemo_default, HMC(1, 0.1, 1))

# Breaks
sample(gdemo_default, HMC(2, 0.1, 1))
