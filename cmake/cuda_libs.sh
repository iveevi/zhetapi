SOURCES='source/cuda/nvarena.cu'
# SOURCES=$SOURCES' source/cuda/essentials.cu'

nvcc --ptxas-options=-v --compiler-options '-fPIC -I engine' -o bin/libzhpcuda.so --shared $SOURCES
