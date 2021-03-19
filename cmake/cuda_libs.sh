echo COMPILING CUDA LIBS

SOURCES=source/cuda/nvarena.cu

nvcc --ptxas-options=-v --compiler-options '-fPIC -I engine' -o bin/libzhpcuda.so --shared $SOURCES
