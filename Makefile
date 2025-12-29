CXX = g++
CXXFLAGS = -mavx512f -mavx512vl -march=native -O3 -ftree-vectorize -funroll-loops

SRC_IMG = ./image_processing/main.cpp
OUT_IMG = ./image_processing/program
SRC_NN = ./neural_network/main.cpp
OUT_NN = ./neural_network/program
SRC_DOT = ./dot_product/main.cpp
OUT_DOT = ./dot_product/program


$(OUT_IMG): $(SRC_IMG)
	$(CXX) $(CXXFLAGS) $(SRC_IMG) -o $(OUT_IMG)

$(OUT_NN): $(SRC_NN)
	$(CXX) $(CXXFLAGS) $(SRC_NN) -o $(OUT_NN)

$(OUT_DOT): $(SRC_DOT)
	$(CXX) $(CXXFLAGS) $(SRC_DOT) -o $(OUT_DOT)

image: $(OUT_IMG)
nn: $(OUT_NN)
dot: $(OUT_DOT)

clean:
	rm $(OUT_IMG) $(OUT_NN) $(OUT_DOT)

.PHONY: rm
