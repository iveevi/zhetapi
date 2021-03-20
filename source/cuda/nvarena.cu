#include <cuda/nvarena.cuh>

#include <iostream>

namespace zhetapi {

// Allocate per megabyte
NVArena::NVArena(size_t mb)
{
	size_t bytes = mb << 20;

	cudaMalloc(&__pool, bytes);

	__cuda_check_error();
}

NVArena::~NVArena()
{
	cudaFree(__pool);
}

}
