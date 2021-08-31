CC := g++

SRCDIR := src
BUILDDIR := build
TARGET1 := lib/libbounds.a
TARGET2 := test/testBounds

AR = ar rcs

SRCEXT := cpp
SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
CFLAGS := -Wall -Wno-unused-result -Wno-unused-function -std=c++14 -O3
LIB := -ldl -lm
INC := -I include

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(OBJECTS)
	@mkdir -p lib
	$(AR) $@ $^

$(TARGET2): test/testBounds.cpp $(TARGET1)
	$(CC) $^ -o $(TARGET2) $(LIB) $(INC)


$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT) #test/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " $(CC) $(CFLAGS) $(INC) -c -o $@ $<"; $(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	@echo " Cleaning...";
	$(RM) -r $(BUILDDIR) $(TARGET1) $(TARGET2)

.PHONY: clean
