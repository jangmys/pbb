###############################################################################
INCLUDE_DIR += include ../common/include
LIBRARY_DIR +=
LDFLAGS += $(addprefix -L,$(LIBRARY_DIR))

BUILD=./build/
#BINDIR=./bin/
SRCDIR=./src/
SRCDIR2=../common/src/
HDRS=./include/


DEBUG = -g -pg -ftrapv
LIB=-ldl -lpthread -lm -lgmp -lgmpxx -lstdc++ -L../bounds/lib -lbounds -L../heuristic -lig
OPT = -Wall -pedantic -Wno-unused-function -Wno-unused-result -O3 -g #$(DEBUG)

CXX=mpicxx

CPPFLAGS = -std=c++11
CPPFLAGS += $(addprefix -I,$(INCLUDE_DIR))

OPT += $(LDFLAGS) $(LIB) $(CPPFLAGS)

# BOUNDS=bound_abstract bound_fsp_strong bound_fsp_weak bound_nqueens bound_test
# INST=instance_flowshop instance_vrf instance_nqueens

FILES=main master worker worker_mc communicator works work interval fact_work weights
FFILES=arguments pbab solution subproblem ttime

OOBJS=$(addprefix $(BUILD), $(addsuffix .o, $(FFILES))) $(addprefix $(BUILD), $(addsuffix .o, $(FILES))) ../multicore/build/matrix_controller.o ../multicore/build/thread_controller.o ../multicore/build/bbthread.o ../multicore/build/ivmthread.o ../multicore/build/sequentialbb.o ../multicore/build/ivm.o ../multicore/build/ivm_bound.o ../beamSearch/build/treeheuristic.o ../beamSearch/build/tree.o
#../multicoreLL/build/treeheuristic.o ../multicoreLL/build/tree.o

bb: $(OOBJS)
	$(CXX) $(OOBJS) -o $@ $(OPT)

$(BUILD)main.o:	$(SRCDIR)main.cpp
	@mkdir -p $(BUILD)
	$(CXX) -o $@ -c $< $(OPT)

$(BUILD)%.o: $(SRCDIR)%.cpp $(HDRS)%.h
	@mkdir -p $(BUILD)
	$(CXX) $(CFLAGS) -c $< -o $@ $(OPT)

$(BUILD)%.o: $(SRCDIR2)%.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CFLAGS) -c $< -o $@ $(OPT)

clean:
	rm -r build bb
