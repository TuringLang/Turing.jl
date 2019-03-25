#!/bin/bash
echo PWD=$PWD

RELEASES=(
    x86_64-w64-mingw32@v1_1@http://cxan.kdr2.com/julia/julia-1.1.0-win64.tar.gz
    # x86_64-w64-mingw32@v1_0@http://mlg.eng.cam.ac.uk/hong/julia/julia-1.0.0-win64.tar.gz
    # x86_64-w64-mingw32@v1_1@http://mlg.eng.cam.ac.uk/hong/julia/julia-1.1.0-win64.tar.gz
    # i686-w64-mingw32@v1_0@http://mlg.eng.cam.ac.uk/hong/julia/julia-1.0.0-win32.tar.gz
    # i686-w64-mingw32@v1_1@http://mlg.eng.cam.ac.uk/hong/julia/julia-1.1.0-win32.tar.gz
)

git clone https://github.com/TuringLang/Turing.jl.git

for RELEASE in ${RELEASES[@]}; do
    echo ============
    echo Working on $RELEASE
    echo ============
    REL_TARGET=$(echo $RELEASE | cut -d@ -f1)
    TARBALL=$(echo $RELEASE | cut -d/ -f5)
    JULIA_VERSION=$(echo $RELEASE | cut -d@ -f2)
    INSTALL_DIR=$(echo $TARBALL | sed 's/.tar.gz$//')
    export JULIA_DEPOT_PATH=$INSTALL_DIR-DEPOT
    curl -O $(echo $RELEASE | cut -d@ -f3)
    tar xzvf $TARBALL > /dev/null
    echo ============
    $INSTALL_DIR/bin/julia --project=. -e 'using Pkg; Pkg.add(PackageSpec(path="../Turing.jl")); Pkg.test("Turing");'
done
