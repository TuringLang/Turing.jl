using .RandomVariables: VarInfo, TypedVarInfo, Metadata, setval!, settrans!
using MCMCChains: Chains
using Turing: PriorContext, LikelihoodContext, SampleFromPrior

macro logprob_str(str)
	expr1, expr2 = get_exprs(str)
    return :(Turing.Core.logprob($expr1, $expr2)) |> esc
end
macro prob_str(str)
	expr1, expr2 = get_exprs(str)
    return :(exp.(Turing.Core.logprob($expr1, $expr2))) |> esc
end

function get_exprs(str::String)
	ind = findfirst(isequal('|'), str)
	str1 = str[1:(ind - 1)]
	str2 = str[(ind + 1):end]

	expr1 = Meta.parse("($str1,)")
	expr1 = Expr(:tuple, expr1.args...)

	expr2 = Meta.parse("($str2,)")
	expr2 = Expr(:tuple, expr2.args...)

	return expr1, expr2
end

function logprob(ex1, ex2)
	@assert isdefined(ex2, :model)
	modelgen = ex2.model
	ptype = probtype(ex1, ex2, modelgen, modelgen.defaults)
	vi = isdefined(ex2, :varinfo) ? ex2.varinfo : nothing
	if ptype isa Val{:prior}
		return logprior(ex1, modelgen, vi)
	elseif ptype isa Val{:likelihood}
		return loglikelihood(ex1, ex2, modelgen, vi)
	end
end

function probtype(
	ntl::NamedTuple{namesl},
	ntr::NamedTuple{namesr},
	modelgen::ModelGen{args},
	defaults::NamedTuple{defs},
) where {namesl, namesr, args, defs}
	basic_namesr = namesr == (:model,) || namesr == (:model, :varinfo)
	@inline valid_arg(arg) = arg in namesl || arg in namesr || (arg in defs) && 
		!(getfield(defaults, arg) isa Missing)

	valid_args = all(valid_arg.(args))
	# Uses the default values for model arguments not provided.
	# If no default value exists, use `nothing`.
	if basic_namesr
		return Val(:prior)
	# Uses the default values for model arguments not provided.
	# If no default value exists or the default value is missing, then error.
	elseif valid_args
		return Val(:likelihood)
	else
		for arg in args
			if !valid_arg(args)
				throw(ArgumentError(missing_arg_error_msg(arg)))
			end
		end
	end
end

missing_arg_error_msg(arg) = """Variable $arg is not defined and has no default value, or its default value is `missing`. Please make sure all the variables are defined or have a default value other than `missing`."""

function logprior(
	left::NamedTuple,
	modelgen::ModelGen,
	_vi::Union{Nothing, VarInfo},
)
	# For model args on the LHS of |, use their passed value but add the symbol to 
	# model.missings. This will lead to an `assume`/`dot_assume` call for those variables.
	# Let `p::PriorContext`. If `p.vars` is `nothing`, `assume` and `dot_assume` will use 
	# the values of the random variables in the `VarInfo`. If `p.vars` is a `NamedTuple` 
	# or a `Chain`, the value in `p.vars` is input into the `VarInfo` and used instead.

	# For model args not on the LHS of |, if they have a default value, use that, 
	# otherwise use `nothing`. This will lead to an `observe`/`dot_observe`call for 
	# those variables.
	# All `observe` and `dot_observe` calls are no-op in the PriorContext

	# When all of model args are on the lhs of |, this is also equal to the logjoint.
	args, missing_vars = get_prior_model_args(left, modelgen, modelgen.defaults)
	model = get_model(modelgen, args, missing_vars)
	vi = _vi === nothing ? VarInfo(deepcopy(model), PriorContext()) : _vi
	foreach(keys(vi.metadata)) do n
		@assert n in keys(left) "Variable $n is not defined."
	end
	model(vi, SampleFromPrior(), PriorContext(left))
	return vi.logp
end
@generated function get_prior_model_args(
	left::NamedTuple{namesl},
	modelgen::ModelGen{args},
	defaults::NamedTuple{default_args},
) where {namesl, args, default_args}
	exprs = []
	missing_args = []
	foreach(args) do arg
		if arg in namesl
			push!(exprs, :($arg = deepcopy(left.$arg)))
			push!(missing_args, arg)
		elseif arg in default_args
			push!(exprs, :($arg = defaults.$arg))
		else
			push!(exprs, :($arg = nothing))
		end
	end
	missing_vars = :(Val{($missing_args...,)}())
	length(exprs) == 0 && :(NamedTuple(), $missing_vars)
	return :(($(exprs...),), $missing_vars)
end

function get_model(modelgen, args, missing_vars)
	_model = modelgen(; args...)
	return Turing.Model(_model.f, args, missing_vars)
end

function loglikelihood(
	left::NamedTuple,
	right::NamedTuple,
	modelgen::ModelGen,
	_vi::Union{Nothing, VarInfo},
)
	# Pass namesl to model constructor, remaining args are missing
	args, missing_vars = get_like_model_args(left, right, modelgen, modelgen.defaults)
	model = get_model(modelgen, args, missing_vars)
	vi = _vi === nothing ? VarInfo(deepcopy(model)) : _vi
	if isdefined(right, :chain)
		# Element-wise likelihood for each value in chain
		ctx = LikelihoodContext()
		return map(1:length(right.chain)) do i
			c = right.chain[i]
			_setval!(vi, c)
			model(vi, SampleFromPrior(), ctx)
			vi.logp
		end
	else
		# Likelihood without chain
		# Rhs values are used in the context
		ctx = LikelihoodContext(right)
		model(vi, SampleFromPrior(), ctx)
		return vi.logp
	end
end
@generated function get_like_model_args(
	left::NamedTuple{namesl},
	right::NamedTuple{namesr},
	modelgen::ModelGen{args},
	defaults::NamedTuple{default_args},
) where {namesl, namesr, args, default_args}
	exprs = []
	missing_args = []
	foreach(args) do arg
		if arg in namesl
			push!(exprs, :($arg = left.$arg))
		elseif arg in namesr
			push!(exprs, :($arg = right.$arg))
			push!(missing_args, arg)
		elseif arg in default_args
			push!(exprs, :($arg = defaults.$arg))
		else
			throw("This point should not be reached. Please open an issue in the Turing.jl repository.")
		end
	end
	missing_vars = :(Val{($missing_args...,)}())
	length(exprs) == 0 && :(NamedTuple(), $missing_vars)
	return :(($(exprs...),), $missing_vars)
end

_setval!(vi::TypedVarInfo, c::Chains) = _setval!(vi.metadata, vi, c)
@generated function _setval!(md::NamedTuple{names}, vi, c) where {names}
	return Expr(:block, map(names) do n
        quote
            for vn in md.$n.vns
                val = copy.(vec(c[string(vn)].value))
                setval!(vi, val, vn)
                settrans!(vi, false, vn)
            end
        end
	end...)
end
