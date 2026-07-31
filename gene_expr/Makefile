CXX ?= /usr/bin/g++
CONDA_PREFIX = /home/francesco/anaconda3/envs/gt

CXXFLAGS = -O3 -fopenmp -std=gnu++17 -Wall -fPIC -pthread \
	-I$(CONDA_PREFIX)/include \
	-I$(CONDA_PREFIX)/lib/python3.12/site-packages/graph_tool/include \
	-I$(CONDA_PREFIX)/lib/python3.12/site-packages/graph_tool/include/boost-workaround \
	-I$(CONDA_PREFIX)/lib/python3.12/site-packages/graph_tool/include/pcg-cpp \
	-I$(CONDA_PREFIX)/lib/python3.12/site-packages/cairo/include \
	-I$(CONDA_PREFIX)/include/python3.12 \
	-I$(CONDA_PREFIX)/lib/python3.12/site-packages/numpy/_core/include \
	-DBOOST_ALLOW_DEPRECATED_HEADERS

LDFLAGS = -shared \
	-L$(CONDA_PREFIX)/lib \
	-L$(CONDA_PREFIX)/lib/python3.12/site-packages/graph_tool \
	-L$(CONDA_PREFIX)/lib/python3.12/site-packages/graph_tool/inference \
	-Wl,-rpath,$(CONDA_PREFIX)/lib:$(CONDA_PREFIX)/lib/python3.12/site-packages/graph_tool:$(CONDA_PREFIX)/lib/python3.12/site-packages/graph_tool/inference \
	-lboost_python312 -lboost_context -lgraph_tool_core -lgraph_tool_inference

TARGET = libgene_expr_ext.so
SRCS = gene_expr_ext.cc

all: $(TARGET)

$(TARGET): $(SRCS) gene_expr_ext.hh
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)
