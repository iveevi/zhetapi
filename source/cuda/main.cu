// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>

// Engine headers
#include <cuda/network.cuh>

#include <std/activation_classes.hpp>
#include <std/optimizer_classes.hpp>

#include <dataset.hpp>

// Defines
#define BENCH

using namespace std;
using namespace zhetapi;

// Main function
int main()
{
	srand(clock());

#ifdef BENCH
	
	ml::NeuralNetwork <double> ::TrainingStatistics ts1;
	ml::NeuralNetwork <double> ::TrainingStatistics ts2;

	const int size = 1;
	const int mlen = 500;
	const int iters = 10;

	ofstream data("data/gpu-cpu-times.dat");

	for (size_t len = 1; len <= mlen; len++) {
		DataSet <double> ins;
		DataSet <double> outs;
		
		for (int i = 0; i < size; i++) {
			ins.push_back(Vector <double> (len,
				[] __host__ __device__ (size_t i) {
					return 2 * (rand()/((double) RAND_MAX)) - 1.0;
				}
			));
			
			outs.push_back(Vector <double> (len,
				[] __host__ __device__ (size_t i) {
					return 2 * (rand()/((double) RAND_MAX)) - 1.0;
				}
			));
		}

		ml::NeuralNetwork <double> model ({
			{len, new zhetapi::ml::Linear <double> ()},
			{len, new zhetapi::ml::Sigmoid <double> ()},
			{len, new zhetapi::ml::ReLU <double> ()},
			{len, new zhetapi::ml::ReLU <double> ()}
		}, []() {return 0.5 - (rand()/(double) RAND_MAX);});

		auto crit = [] __device__ (zhetapi::Vector <double> actual,
				zhetapi::Vector <double> expected) {
			return actual == expected;
		};

		ml::Optimizer <double> *opt = new zhetapi::ml::MeanSquaredError <double> ();

		model.randomize();

		model.set_cost(opt);
		
		double gpu_ktime = 0;
		double gpu_ftime = 0;
		double cpu_ktime = 0;

		for (int j = 0; j < iters; j++) {
			ts1 = model.cuda_epochs(ins, outs, 1, 100, 0.1, crit, false);
			ts2 = model.epochs(ins, outs, 1, 100, 0.1, false);

			gpu_ktime += ts1.__kernel_time;
			gpu_ftime += ts1.__full_time;
			cpu_ktime += ts2.__kernel_time;
		}

		gpu_ktime /= ((double) iters);
		gpu_ftime /= ((double) iters);
		cpu_ktime /= ((double) iters);

		data << len
			<< "\t" << gpu_ktime
			<< "\t" << gpu_ftime
			<< "\t" << cpu_ktime << endl;

		// Free resources
		delete opt;
	}

#else

	const int size = 1;
	const int len = 2;
	
	DataSet <double> ins;
	DataSet <double> outs;
	
	for (int i = 0; i < size; i++) {
		ins.push_back(Vector <double> (len,
			[] __host__ __device__ (size_t i) {
				return 2 * (rand()/((double) RAND_MAX)) - 1.0;
			}
		));
		
		outs.push_back(Vector <double> (len,
			[] __host__ __device__ (size_t i) {
				return 2 * (rand()/((double) RAND_MAX)) - 1.0;
			}
		));
	}

	ml::NeuralNetwork <double> model ({
		{len, new zhetapi::ml::Linear <double> ()},
		{len, new zhetapi::ml::Sigmoid <double> ()},
		{len, new zhetapi::ml::ReLU <double> ()},
		{len, new zhetapi::ml::ReLU <double> ()}
	}, []() {return 0.5 - (rand()/(double) RAND_MAX);});
	
	auto crit = [] __device__ (zhetapi::Vector <double> actual,
			zhetapi::Vector <double> expected) {
		return actual == expected;
	};

	ml::Optimizer <double> *opt = new zhetapi::ml::MeanSquaredError <double> ();

	model.randomize();

	model.set_cost(opt);
			
	model.cuda_epochs(ins, outs, 1, 100, 0.1, crit, true);
	model.epochs(ins, outs, 1, 100, 0.1, true);

#endif

}
