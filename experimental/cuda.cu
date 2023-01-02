#include <iostream>

#include <jitify/jitify.hpp>

const std::string source = R"(
__global__
void kernel_add(float *dst, float *x, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = 0; i < N; i += stride)
		dst[idx] += x[idx];
}
)";

int main()
{
	std::cout << "CUDA JIT Testing..." << std::endl;

	jitify::JitCache kernel_cache;
	jitify::Program program = kernel_cache.program(source);

	std::cout << "Compiling kernel..." << std::endl;

	int N = 1024;

	float *dst = new float[N];
	float *x = new float[N];

	for (int i = 0; i < N; i++) {
		dst[i] = i * i - 1;
		x[i] = 1.0f;
	}

	float *d_dst, *d_x;

	cudaMalloc(&d_dst, N * sizeof(float));
	cudaMalloc(&d_x, N * sizeof(float));

	cudaMemcpy(d_dst, dst, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

	delete[] dst;
	delete[] x;

	std::shared_ptr <float []> s_dst(d_dst, cudaFree);
	std::shared_ptr <float []> s_x(d_x, cudaFree);

	jitify::KernelInstantiation kernel = program.kernel("kernel_add").instantiate();

	std::cout << "Launching kernel..." << std::endl;

	kernel.configure(dim3(32), dim3(32)).launch(d_dst, d_x, N);

	float *h_dst = new float[N];

	cudaMemcpy(h_dst, d_dst, N * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "Results:" << std::endl;
	for (int i = 0; i < N; i++)
		std::cout << h_dst[i] << " ";
	std::cout << std::endl;
}
