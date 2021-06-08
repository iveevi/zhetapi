#include <timer.hpp>
#include <tensor.hpp>
#include <vector.hpp>
#include <cuda/nvarena.cuh>

#include <iostream>

using namespace zhetapi;
using namespace std;

// Simulation class

void io_block()
{
	static std::string input;

	cout << "Blocking (until enter) ";
	getline(cin, input);
}

template <class T>
__global__
void double_kernel(Vector <T> *vptr)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;

	// size_t incr = blockDim.x;
	if (i < vptr->size())
		vptr->get(i) *= 2;
}

template <class T>
__global__
void add_kernel(Vector <T> *vptr1, Vector <T> *vptr2, Vector <T> *vptr3)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;

	while (i < vptr1->size()) {
		vptr3->get(i) = vptr1->get(i) + vptr2->get(i);

		i += stride;
	}
}

int main()
{
	NVArena arena(4096);

	Vector <double> vec1(10, 5);
	Vector <double> vec2(10, 12.4);

	cout << "vec = " << vec1 << endl;

	Vector <double> *hc1 = vec1.cuda_half_copy(&arena);
	Vector <double> *fc1 = hc1->cuda_full_copy(&arena);

	Vector <double> *hc2 = vec2.cuda_half_copy(&arena);
	Vector <double> *fc2 = hc2->cuda_full_copy(&arena);

	double_kernel <<<10, 1>>> (fc1);
	add_kernel <<<10, 1>>> (fc1, fc2, fc1);

	vec1.cuda_read(hc1);
	cout << "post-vec = " << vec1 << endl;

	arena.free(fc1);
	arena.free(fc2);

	delete hc1;
	delete hc2;

	arena.show_mem_map();
}
