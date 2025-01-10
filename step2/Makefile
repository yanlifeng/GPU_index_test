# Compiler and flags
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++17 -O3 -w -g
NVCCFLAGS = -std=c++17 -O3 -w -arch=sm_86 -g


# Directories
SRC_DIR = .
OBJ_DIR = obj

# Sources and headers
CPP_SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
CU_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
HEADERS = $(wildcard $(SRC_DIR)/*.hpp $(SRC_DIR)/kseq++/*.h $(SRC_DIR)/RabbitFX/*.h)

# Object files
CPP_OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))
CU_OBJECTS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SOURCES))

# Target executable
TARGET = main

# Rules
all: $(TARGET)

$(TARGET): $(CPP_OBJECTS) $(CU_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lz

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean

