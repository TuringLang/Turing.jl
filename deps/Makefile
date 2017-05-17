CC=gcc
JL_SHARE = $(shell julia -e 'print(joinpath(JULIA_HOME,Base.DATAROOTDIR,"julia"))')
CFLAGS   += $(shell julia $(JL_SHARE)/julia-config.jl --cflags)
CXXFLAGS += $(shell julia $(JL_SHARE)/julia-config.jl --cflags)
LDFLAGS  += $(shell julia $(JL_SHARE)/julia-config.jl --ldflags)
LDLIBS   += $(shell julia $(JL_SHARE)/julia-config.jl --ldlibs)

CFLAGS   += $(shell julia -e 'print("-std=gnu99 ")')$
CFLAGS   += $(shell julia -e 'if is_windows() && Int == Int32 print("-march=pentium4 ") end')$

task: task.c
	$(CC) $(CFLAGS) -O2 -shared -fPIC task.c $(LDFLAGS) $(LDLIBS) -o libtask.$(LIBEXT)

platform=$(shell uname)
ifeq ($(platform),Linux)
        LIBEXT=so
else ifeq ($(platform),Darwin)
        LIBEXT=dylib
else
        LIBEXT=dll
endif

clean:
	-rm libtask.$(LIBEXT)

.PHONY: \
	task

