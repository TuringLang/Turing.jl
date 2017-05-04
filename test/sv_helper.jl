# Load data
# NOTE: put the data file in the same path of this file
readsvdata() = readcsv("sv_nuts.data")[:,2][2:end]
