# Load data
# NOTE: put the data file in the same path of this file

readsvdata() = begin
  y = readcsv(TPATH*"/example-models/nuts-paper/sv_nuts.data")[:,2][2:end]
  y = Float64[map(i -> isa(i, Real) ? i : 0.1, y)...]
end
