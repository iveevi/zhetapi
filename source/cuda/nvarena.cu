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

void *NVArena::alloc(size_t bytes)
{
	using namespace std;
	// Case where __flist is empty
	if (__flist.empty()) {
		cout << "empty flist..." << endl;

		// Assign to the free list
		__flist[__pool] = bytes;

		return __pool;
	}
	
	// Get the last block
	auto last = __flist.rbegin();

	// Allocation strategy: allocate from the end of the arena
	void *laddr = last->first + last->second;

	cout << "laddr = " << laddr << endl;

	// Assign to the free list
	__flist[laddr] = bytes;

	return laddr;
}

}
