#include <cuda/network.cuh>

int main()
{
	zhetapi::cuda::kernel <<<1, 12>>> (4, 5);
}
