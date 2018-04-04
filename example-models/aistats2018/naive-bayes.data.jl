using HDF5, JLD

const nbmnistdata = load(Pkg.dir("Turing")*"/example-models/aistats2018/mnist-10000-40.data")["data"]