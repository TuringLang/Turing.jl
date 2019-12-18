function getexprs(expr)
	str = string(expr)
	ind = findfirst(isequal('|'), str)
	str1 = str[1:(ind - 1)]
	str2 = str[(ind + 1):end]
	strs1 = split(str1, ',')
	strs2 = split(str2, ',')
	expr1 = Expr(:tuple, Meta.parse.(strs1)...)
	expr2 = Expr(:tuple, Meta.parse.(strs2)...)
	return expr1, expr2
end

macro 