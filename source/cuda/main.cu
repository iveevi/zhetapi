// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>

// Engine headers
#include <cuda/network.cuh>

#include <std/activation_classes.hpp>
#include <std/optimizer_classes.hpp>

#include <dataset.hpp>

using namespace std;
using namespace zhetapi;

// For profiling
__global__ void reset_gpu_counters()
{
	gpu_tensor_copies = 0;
	gpu_matrix_copies = 0;
	gpu_vector_copies = 0;
}

__global__ void print_gpu_counters()
{
	printf("GPU-tensor (objects)\t%lu\n", gpu_tensor_copies);
	printf("GPU-matrix (objects)\t%lu\n", gpu_matrix_copies);
	printf("GPU-vector (objects)\t%lu\n", gpu_vector_copies);
}

void reset_counters()
{
	cpu_tensor_copies = 0;
	cpu_matrix_copies = 0;
	cpu_vector_copies = 0;
	
	reset_gpu_counters <<<1, 1>>> ();

	cudaDeviceSynchronize();
}

void print_counters()
{
	printf("\n========================\n");
	printf("CPU-tensor (objects)\t%lu\n", cpu_tensor_copies);
	printf("CPU-matrix (objects)\t%lu\n", cpu_matrix_copies);
	printf("CPU-vector (objects)\t%lu\n", cpu_vector_copies);
	
	printf("------------------------\n");

	print_gpu_counters <<<1, 1>>> ();

	cudaDeviceSynchronize();
}

// Main function
int main()
{
	// srand(clock());

	const int size = 10;
	
	DataSet <double> ins;
	DataSet <double> outs;
	
	for (int i = 0; i < size; i++) {
		ins.push_back({(i + 1)/23.0, (i - 0.5)/23.0});
		outs.push_back({rand()/(RAND_MAX * 0.1), rand()/(RAND_MAX * 0.1)});
	}

	ml::NeuralNetwork <double> model ({
		{2, new zhetapi::ml::Linear <double> ()},
		{5, new zhetapi::ml::Sigmoid <double> ()},
		{5, new zhetapi::ml::ReLU <double> ()},
		{2, new zhetapi::ml::ReLU <double> ()}
	}, []() {return 0.5 - (rand()/(double) RAND_MAX);});

	auto crit = [] __device__ (zhetapi::Vector <double> actual,
			zhetapi::Vector <double> expected) {
		return actual == expected;
	};

	ml::Optimizer <double> *opt = new zhetapi::ml::MeanSquaredError <double> ();

	model.randomize();

	model.set_cost(opt);

	cout << "GPU Training..." << endl;
	model.cuda_epochs <decltype(crit), 1, 10> (ins, outs, 1, 10, 0.1, crit, true);

	print_counters();
	reset_counters();

	cout << endl << "CPU Training..." << endl;
	model.epochs <10> (ins, outs, 1, 10, 0.1, true);
	
	print_counters();
	reset_counters();

	// Free resources
	delete opt;
}
