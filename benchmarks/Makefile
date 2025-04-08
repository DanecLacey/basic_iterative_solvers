include ../config.mk

# apply solver parameters
CXXFLAGS += -DSIMD_LENGTH=$(SIMD_LENGTH)
CXXFLAGS += -DINIT_X_VAL=$(INIT_X_VAL)
CXXFLAGS += -DB_VAL=$(B_VAL)

# compiler options
ifeq ($(COMPILER),gcc)
  CXX       = g++
  OPT_LEVEL = -O3
  OPT_ARCH  = -march=native
  CXXFLAGS += $(OPT_LEVEL) -Wall -fopenmp $(OPT_ARCH) -std=$(CPP_VERSION)
endif

ifeq ($(COMPILER),icc)
  CXX       = icpc
  OPT_LEVEL = -O3
  OPT_ARCH  = -xhost
  CXXFLAGS += $(OPT_LEVEL) -Wall -fopenmp $(OPT_ARCH) -std=$(CPP_VERSION)
endif

ifeq ($(COMPILER),icx)
  CXX       = icpx
  OPT_LEVEL = -O3
  OPT_ARCH  = -xhost
  AVX512_fix= -Xclang -target-feature -Xclang +prefer-no-gather
  CXXFLAGS += $(OPT_LEVEL) -Wall -fopenmp $(AVX512_fix) $(OPT_ARCH) -std=$(CPP_VERSION)
endif

ifeq ($(COMPILER),clang)
  CXX       = clang++
  OPT_LEVEL = -Ofast
  OPT_ARCH  = -march=native
  CXXFLAGS += $(OPT_LEVEL) -std=$(CPP_VERSION) -Wall -fopenmp $(OPT_ARCH)
endif

ifeq ($(DEBUG_MODE),1)
  DEBUGFLAGS += -g -DDEBUG_MODE
endif

ifeq ($(DEBUG_MODE_FINE),1)
  DEBUGFLAGS += -g -DDEBUG_MODE -DDEBUG_MODE_FINE
endif

ifeq ($(USE_LIKWID),1)
  ifeq ($(LIKWID_INC),)
    $(error USE_LIKWID selected, but no include path given in LIKWID_INC)
  endif
  ifeq ($(LIKWID_LIB),)
    $(error USE_LIKWID selected, but no library path given in LIKWID_LIB)
  endif
	LIBS += $(LIKWID_LIB)
	INCLUDES += $(LIKWID_INC)
  CXXFLAGS  += -DUSE_LIKWID -DLIKWID_PERFMON -llikwid
endif

# Target executable
TARGET=run_benchmarks

LINKS := $(filter-out ../main.o, $(wildcard ../*.o))

.PHONY: all clean

all: $(TARGET)

clean:
	rm -f $(TARGET) run_benchmarks.o

# Link the object files to create the executable
$(TARGET): run_benchmarks.o timing.o ${LINKS}
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ run_benchmarks.o timing.o ${LINKS} ${LIBS} ${INCLUDES}

# Compile run_benchmarks.c into an object file
run_benchmarks.o: run_benchmarks.cpp
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) ${LINKS} ${INCLUDES} ${LIBS} -c -o $@  $<

# Compile run_benchmarks.c into an object file
timing.o: timing.c
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) -c -o $@ $<