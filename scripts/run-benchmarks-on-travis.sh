#!/bin/bash
set -v

BASE_BRANCH=master

if [[ $TRAVIS == true ]]; then
    if [[ $TRAVIS_PULL_REQUEST == true ]]; then
        CURRENT_BRANCH=$TRAVIS_PULL_REQUEST_BRANCH
        exit 0 # avoid double run on commit and PR
    else
        CURRENT_BRANCH=$TRAVIS_BRANCH
    fi
else
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
fi
        
COMMIT_MSG=$(git show -s --format="%s")

SANTI_BR_NAME=$(echo $CURRENT_BRANCH | sed 's/\W/_/g')
COMMIT_SHA=$(git rev-parse  HEAD)
TIME=$(date +%Y%m%d%H%M%S)
BM_JOB_NAME="BMCI-${SANTI_BR_NAME}-${COMMIT_SHA:0:7}-${TIME}"

if [[ $COMMIT_MSG != *"[bm]"* ]]; then
    echo "skipping the benchmark jobs."
    exit 0
fi

git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
git fetch --all --unshallow

git clone https://github.com/TuringLang/TuringBenchmarks.git ../TuringBenchmarks
# Notice: uncomment the following line to use travis-ci branch of TuringBenchmarks
# git -C ../TuringBenchmarks checkout -b travis-ci origin/travis-ci

unset JULIA_PROJECT
cat > pre.jl <<EOF
using Pkg
# Pkg.instantiate()
try pkg"develop ." catch end
try pkg"develop ." catch end
try pkg"build Turing" catch end
using Turing
try pkg"develop ../TuringBenchmarks" catch end
try pkg"develop ../TuringBenchmarks" catch end
pkg"add SpecialFunctions"
using TuringBenchmarks
EOF

julia pre.jl

cat > run.jl <<EOF
using TuringBenchmarks.Runner
Runner.run_bm_on_travis("$BM_JOB_NAME", ("master", "$CURRENT_BRANCH"), "$COMMIT_SHA")
EOF

cat run.jl
julia run.jl
