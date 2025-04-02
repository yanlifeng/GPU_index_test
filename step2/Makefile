# Compiler
CXX = nvcc
NVCC = nvcc

# Include path
INCLUDES = -I./ -I/home/user_home/ylf/someGit/gallatin/include/

# Base flags
BASEFLAGS = -std=c++17 -arch=sm_86 -w -rdc=true $(INCLUDES)

# Release/Debug flags
RELEASE_FLAGS = -O2
DEBUG_FLAGS   = -O2 -G

# Directories
SRC_DIR = .
OBJ_DIR = obj

# Sources
CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
CU_SOURCES  = $(wildcard $(SRC_DIR)/*.cu)
HEADERS     = $(wildcard $(SRC_DIR)/*.hpp $(SRC_DIR)/kseq++/*.h $(SRC_DIR)/RabbitFX/*.h)

# Object file lists
RELEASE_CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/release_%.o,$(CPP_SOURCES))
RELEASE_CU_OBJECTS  = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/release_%.o,$(CU_SOURCES))
DEBUG_CPP_OBJECTS   = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/debug_%.o,$(CPP_SOURCES))
DEBUG_CU_OBJECTS    = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/debug_%.o,$(CU_SOURCES))

# Targets
RELEASE_TARGET = main_release
DEBUG_TARGET   = main_debug

# Default target
all: $(RELEASE_TARGET) $(DEBUG_TARGET)

# Build release executable
$(RELEASE_TARGET): $(RELEASE_CPP_OBJECTS) $(RELEASE_CU_OBJECTS)
	$(NVCC) $(BASEFLAGS) $(RELEASE_FLAGS) -o $@ $^ -lz

# Build debug executable
$(DEBUG_TARGET): $(DEBUG_CPP_OBJECTS) $(DEBUG_CU_OBJECTS)
	$(NVCC) $(BASEFLAGS) $(DEBUG_FLAGS) -o $@ $^ -lz

# Release object build rules
$(OBJ_DIR)/release_%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(BASEFLAGS) $(RELEASE_FLAGS) -c $< -o $@

$(OBJ_DIR)/release_%.o: $(SRC_DIR)/%.cu $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(BASEFLAGS) $(RELEASE_FLAGS) -c $< -o $@

# Debug object build rules
$(OBJ_DIR)/debug_%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(BASEFLAGS) $(DEBUG_FLAGS) -c $< -o $@

$(OBJ_DIR)/debug_%.o: $(SRC_DIR)/%.cu $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(BASEFLAGS) $(DEBUG_FLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(OBJ_DIR) main_debug main_release

.PHONY: all clean

