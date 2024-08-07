module AquaTests

using Aqua: Aqua
using Turing

# We test ambiguities separately because it catches a lot of problems
# in dependencies but we test it for Turing.
Aqua.test_ambiguities([Turing])
Aqua.test_all(Turing; ambiguities=false)


end
