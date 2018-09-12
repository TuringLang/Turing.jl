using Documenter, Turing

ENV["DOCUMENTER_DEBUG"] = true

# ?? TRAVIS_REPO_SLUG       = ""
# ??   should match "github.com/cpfiffer/Turing.jl.git" (kwarg: repo)
# ?? TRAVIS_PULL_REQUEST    = ""
# ??   deploying if equal to "false"
# ?? TRAVIS_OS_NAME         = ""
# ??   deploying if equal to "linux" (kwarg: osname)
# ?? TRAVIS_JULIA_VERSION   = ""
# ??   deploying if equal to "1.0" (kwarg: julia)
# ?? TRAVIS_BRANCH          = ""
# ?? TRAVIS_TAG             = ""
# ??   deploying if branch equal to "master" (kwarg: latest) or tag is set
# ?? git commit SHA         = 39148d2
# ?? DOCUMENTER_KEY exists  = false
# ?? should_deploy          = false

ENV["TRAVIS_REPO_SLUG"] = "github.com/cpfiffer/Turing.jl.git"
ENV["TRAVIS_PULL_REQUEST"] = false
ENV["TRAVIS_OS_NAME"] = "linux"
ENV["TRAVIS_JULIA_VERSION"] = "1.0"
ENV["TRAVIS_BRANCH"] = "master"
ENV["TRAVIS_TAG"] = ""
ENV["DOCUMENTER_KEY"] = "LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlFcEFJQkFBS0NBUUVBcGt5S0l1K0tMRkNpY0YwNXFUSUZwRzUxN25lcjlDYjJFRFR5S0N2SWVDMXhLSHJoClpOOW9haGhxZStVbENBRVhCbmFqMlBJZWVXTC96NS9KOFVOZ0hzNVJBM2ZxNWlMY3I4N0p2S2kvWXBoeUt1alQKT0tidUl3V1N6b3VvVUg1cjhuRm5HVUVZVjFOS0NpTDZhMGp0b3RyOE1UVU5BVDg2NnRsT0pMR3NaSmhwL2xPUwpWZStzeHJDTFAxbzNZbVkzaWJGVmNZNE1hZit1NXMweHZXVW1xOERncUtDR1dEd2FPVU44S2UvcDJEbkNKbmt4Ck42M05NRnJVVm1QSTl5SnhRcVhsZ0FzRnlWeUxJb0tWdjFHQzJiRWNad1BGRmZpTk93UHlCcHV1eTFnZ0NFLy8KNWtadTV3SjFQV2N6eWZ5akY4TVhUNUt1aFBnNmFsdHd0MVpVOXdJREFRQUJBb0lCQURQZEZRdTJOeHFBLzFnWgp5dnpZaStmdlJ6cGEramJIMjkvTUUyV3gyZkNaQTN5RGJqMythdHNEeUZuaGFmNG9FTjFFTE85ZysxNFNJUVBJCnFydGlydHlNcmJsdU1jVWFSRWRVTDVoMTdGSThod2FZM0I4RCtLclZkeWFSYWFuSTg3T0Y2SWg0dzdXZ1ZTdjYKSXExSTdrdm5EZFh1M0tKeDdOY0hkSlovRHV2RUdEMmNuSTh6TkFjZEJSOGo2MkZGQ1k4VllCY0RtQmFVWnVaeQppVmZKNTJmOWthR1g4N0dwQno3YmtESHV6d1YxQkVXT1Q2MjlCdzJ6bkJ6MFhhUDRtNld0aGsvakZkNTh1b2tuCkhEM2E0dnVveFd6dXBaUmVUdWpYSDVYMXFWMitEUEEwYmFvMm5DcTVnb2NaZ1NDZFhiMDhtSUR0YkhvRXU0b1UKanNueVM5RUNnWUVBMEkrTDRKamhTcmNRWTNzMUs2UEdmY2RmSW9WZ0l2OGtVMTVQalZyKzFSNFdublJzRlV0VwpJQ3E0NEtQVHgzYnRQRmljb2ZsM3Z1TjZjU2JTNzkxWUlpL2YxblQ1cDBGOVcyY2wvQi9qMEM3Smt3b3BSeEsrClY1WVN6YTZZTTFNOFgwNDhQbTZ6cWRVTnB2MUNiN0VuRUE1N2YzaktNRkJWMHZPVS82cjM3aDhDZ1lFQXpDQVoKY1h5Nitja3I1UWZVWnlaMnp1LzlBNEtpYm1nT3dTTnlEeDUxNk91VUVhUmxySS9XeGNRYmVNeFRUczJkZzJFdgp6Rm9ydWF2NG1jbStWT2dwcmZqbEM4R3FlcFJxR2cxdU1XQ0RjWWZHVWFyYjFxbzJqSHFSdUhvUThWTTNneXpxClBKc0hRc0ZNZ01teGpybm1pcGx0TWxkOTQzcFNQOHlsVnViMGppa0NnWUJOd0RYMVZRa29RWGJxbjFRRElOc3UKcU5UZW1icHRVWkRKUTYrRWt4Zk5lNW9UR0hQeXZGTXZTcjRCZ1JIQ21xTjJpOUpZMEJmRUlpblRaUlhlTmpaVgpDR1A2SDBHekNNY2kvQ0U5RE9aeEJnQS93Tm9DbFFwQXZNSGx5K0VSd1VZUWdhb1QzRkUrVHg3MzBoS2ljUGlQCkU5Y1dmb0ZoNFpFZFE1R1lmclJQSVFLQmdRQ0w0ZWR0dFFzd3g2Sno5alNoWjJOOWxrcHQwR1RkZ2lPeVNVY04KZzFOTUJieFhockJDVytQVGJQdGlnYXNKVXJDQmF1VmxoZGwzQ0psNVVoNURjMEcwdmQ3QWVyd0grcExuUFpMbwo5WG0rSUV4UWhPVFlLNWJzRjhhcWc4UCtqSWQ3TmhsaTVONUo2Y3N5YW9WcUNJMHJKODhEODU4S2R6WE1FTUt4ClZkMzdXUUtCZ1FDWTVWc2xkOTN5ek1FTkNrNUN5QnU4YWt2QW1DQnZxN2VXZXBsdEt0OGJNWnJvaFBkVCtMRlkKd2lnU3lydWtBWXl3aHdKOFVINUpmczFXT1VubGN4OFZMSFR3WkVqUU9SUWZnNitVMzEvMTJMR0NxK0NpWlErZwptQ2hVSkVzcEkxdFpYYWIzOEdwYTd4bTVFazVaeUxDSmUxUzhtWHJxRWMxK2gvVDdmbE9CWGc9PQotLS0tLUVORCBSU0EgUFJJVkFURSBLRVktLS0tLQo="

makedocs(
    format = :html,
    sitename = "Turing.jl",
    pages = [
        "Home" => ["index.md",
                   "get-started.md",
                   "advanced.md",
                   "contributing/guide.md",
                   "contributing/style_guide.md",],
        "Tutorials" => ["ex/0_Introduction.md"],
        "A Test" => "ex/misc.md",
        "API" => "api.md"
    ]
)

# deploydocs(
#     repo = "github.com/cpfiffer/Turing.jl.git",
#     target = "build",
#     deps   = nothing,
#     make   = nothing,
#     julia = "1.0"
# )
