# Part of the Parabix Project, under the Open Software License 3.0.
# SPDX-License-Identifier: OSL-3.0

# Module: util

# Compiler flags (common for both debug and release)
CXXFLAGS = -I/path/to/kernel/streamutils
CCFLAGS = -I/path/to/kernel/streamutils

# Debug-specific flags
debug: CXXFLAGS += -DDEBUG -g
debug: CCFLAGS += -DDEBUG -g

# Targets
all: hexlines hashdemo filterbymask

# Rule for hexlines
hexlines: hexlines.cpp
    $(CXX) $(CXXFLAGS) $< -o $@ $(DEPS)

# Rule for hashdemo
hashdemo: hashdemo.cpp
    $(CXX) $(CXXFLAGS) $< -o $@ $(DEPS)

# Rule for filterbymask
filterbymask: filterbymask.cpp
    $(CXX) $(CXXFLAGS) $< -o $@ $(DEPS)

# Clean rule
clean:
    rm -f hexlines hashdemo filterbymask *.o

