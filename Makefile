include config.mk

# apply solver parameters
CXXFLAGS += -DSIMD_LENGTH=$(SIMD_LENGTH)
CXXFLAGS += -DMAX_ITERS=$(MAX_ITERS)
CXXFLAGS += -DTOL=$(TOL)
CXXFLAGS += -DGMRES_RESTART_LEN=$(GMRES_RESTART_LEN)
CXXFLAGS += -DRES_CHECK_LEN=$(RES_CHECK_LEN)
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

ifeq ($(OUTPUT_SPARSITY),1)
  CXXFLAGS += -DOUTPUT_SPARSITY
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

ifeq ($(USE_SCAMAC),1)
  ifeq ($(SCAMAC_INC),)
    $(error SCAMAC_INC selected, but no include path given in SCAMAC_INC)
  endif
  ifeq ($(SCAMAC_LIB),)
    $(error SCAMAC_LIB selected, but no library path given in SCAMAC_LIB)
  endif
  LIBS += $(SCAMAC_LIB)
  INCLUDES += -I$(SCAMAC_INC)
  CXXFLAGS += -DUSE_SCAMAC
endif

basic_iterative_solvers: main.o mmio.o
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) mmio.o main.o -o basic_iterative_solvers $(LIBS) $(INCLUDES)

main.o: main.cpp $(REBUILD_DEPS)
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) -c main.cpp -o main.o  $(LIBS) $(INCLUDES)

mmio.o: mmio.cpp $(REBUILD_DEPS)
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) -c mmio.cpp -o mmio.o

clean:
	-rm *.o
