using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
    "--n_iters"
        arg_type = Int
        default = 2_000
    "--n_adapt"
        arg_type = Int
        default = 500
    "--step_size"               # step size for HMC
        arg_type = Float64
        default = 0.1
    "--step_num"                # step number for HMC
        arg_type = Int
        default = 10
    "--step_len"                # step length for HMCDA
        arg_type = Float64
        default = 0.1
    "--tar_acc_rate"
        arg_type = Float64
        default = 0.85
    "--alg"
        arg_type = String
        default = "NUTS"
    "--model"
        arg_type = String
        default = "gdemo"
    "--pc"
        arg_type = String
        default = "diag"
    "--describe"
        action = :store_true
end

args = parse_args(s; as_symbols=true)

using Turing; Turing.setadbackend(:reverse_diff)

# gdemo is used if not matched
if args[:model] == "m12_2"
    include("m12_2.jl")
else
    @model gdemo(x) = begin
      s ~ InverseGamma(2, 3)
      m ~ Normal(0, sqrt(s))
      x[1] ~ Normal(m, sqrt(s))
      x[2] ~ Normal(m, sqrt(s))
      return s, m
    end

    mf = gdemo([1.5, 2.0])
end

# NUTS is used if not matched
alg = if args[:alg] == "MH"
    Turing.MH(args[:n_iters])
elseif args[:alg] == "HMC"
    Turing.HMC(args[:n_iters], args[:step_size], args[:step_num])
elseif args[:alg] == "HMCDA"
    Turing.HMCDA(args[:n_iters], args[:n_adapt], args[:tar_acc_rate], args[:step_len])
else
    Turing.NUTS(args[:n_iters], args[:n_adapt], args[:tar_acc_rate])
end

# UnitPreConditioner is used if not matched
pc_type = if args[:pc] == "dense"
    Turing.Inference.DensePreConditioner
elseif args[:pc] == "diag"
    Turing.Inference.DiagPreConditioner
else
    Turing.Inference.UnitPreConditioner
end

posterior = if args[:alg] == "MH"
    sample(mf, alg)
else
    sample(mf, alg; pc_type=pc_type)
end

# Compute average ESS of all modelvariables
posterior_summary = MCMCChain.summarystats(posterior)
ess_idx = findall(x -> x == "ESS", posterior_summary.colnames)[1]   # get the index of ESS column
var_idcs = findall(x -> x[1] != '_', posterior_summary.rownames)    # get the indices of all model variables
                                                                    # excluding internal ones
var_ess = posterior_summary.value[var_idcs,ess_idx,1]               # get ESS of all model variables
@info "Average ESS of all model variables)" alg mean(var_ess)

args[:describe] && describe(posterior)

# ┌ Info: Average ESS of all model variables)
# │   alg = MH{Any}(2000, Dict{Symbol,Any}(), Set(Any[]), 0)
# └   mean(var_ess) = 33.7456617236659

# ┌ Info: Average ESS of all model variables)
# │   alg = HMCDA{Turing.Core.FluxTrackerAD,Any}(2000, 500, 0.85, 0.3, Set(Any[]), 0)
# └   mean(var_ess) = 367.31149021291526

# ┌ Info: Average ESS of all model variables)
# │   alg = NUTS{Turing.Core.FluxTrackerAD,Union{}}(2000, 500, 0.85, Set(Union{}[]), 0)
# └   mean(var_ess) = 37.44672051030364
