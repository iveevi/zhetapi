#include <cuda/nvarena.cuh>

#include <iostream>

namespace zhetapi {

// Allocate per megabyte
NVArena::NVArena(size_t mb)
{
	size_t bytes = mb << 20;

	using namespace std;
	cout << "Bytes = " << bytes << endl;

	int *pool;
	cudaMalloc(&pool, bytes);

	cout << "pool = " << pool << endl;

	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
}

NVArena::~NVArena()
{
	cudaFree(__pool);
}

}
