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
// #define BENCH

using namespace std;
using namespace zhetapi;

// Main function
int main()
{
	srand(clock());

#ifdef BENCH
	
	ml::NeuralNetwork <double> ::TrainingStatistics ts1;
	ml::NeuralNetwork <double> ::TrainingStatistics ts2;
	ml::NeuralNetwork <double> ::TrainingStatistics ts3;
	ml::NeuralNetwork <double> ::TrainingStatistics ts4;

	const int size = 10;
	const int mlen = 500;
	const int iters = 10;

	ofstream data("data/gpu-cpu-times.dat");

	for (size_t len = 1; len <= mlen; len++) {
		cout << "Starting len = " << 1 << endl;

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
			{len, new zhetapi::ml::ReLU <double> ()},
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
		double cpug_ktime = 0;
		double cpug_ftime = 0;
		double cpu_ktime = 0;
		double cput_ktime = 0;

		for (int j = 0; j < iters; j++) {
			ml::opt = 1;
			ts1 = model.cuda_epochs(ins, outs, 1, max(len/10, 1L), 0.1, crit, false);
			ml::opt = 0;
			ts2 = model.cuda_epochs(ins, outs, 1, max(len/10, 1L), 0.1, crit, false);
			ts3 = model.epochs(ins, outs, 1, max(len/10, 1L), 0.1, false);
			ts4 = model.epochs <8> (ins, outs, 1, max(len/10, 1L), 0.1, false);

			gpu_ktime += ts1.__kernel_time;

			// Skip the large initial jump (outlier)
			if (j != 0)
				gpu_ftime += ts1.__full_time;

			cpug_ktime += ts2.__kernel_time;
			cpug_ftime += ts2.__full_time;
			cpu_ktime += ts3.__kernel_time;
			cput_ktime += ts4.__kernel_time;
		}

		gpu_ktime /= ((double) iters);
		gpu_ftime /= ((double) (iters - 1));
		cpug_ktime /= ((double) iters);
		cpug_ftime /= ((double) iters);
		cpu_ktime /= ((double) iters);
		cput_ktime /= ((double) iters);

		data << len
			<< "\t" << gpu_ktime
			<< "\t" << gpu_ftime
			<< "\t" << cpug_ktime
			<< "\t" << cpug_ftime
			<< "\t" << cpu_ktime
			<< "\t" << cput_ktime << endl;

		// Free resources
		delete opt;
	}

#else

	const int size = 1;
	
	DataSet <double> ins;
	DataSet <double> outs;
	
	for (int i = 0; i < size; i++) {
		ins.push_back(Vector <double> (2,
			[] __host__ __device__ (size_t i) {
				return 2 * (rand()/((double) RAND_MAX)) - 1.0;
			}
		));
		
		outs.push_back(Vector <double> (4,
			[] __host__ __device__ (size_t i) {
				return 2 * (rand()/((double) RAND_MAX)) - 1.0;
			}
		));
	}

	ml::NeuralNetwork <double> model ({
		{2, new zhetapi::ml::Linear <double> ()},
		{3, new zhetapi::ml::Sigmoid <double> ()},
		{5, new zhetapi::ml::ReLU <double> ()},
		{4, new zhetapi::ml::ReLU <double> ()}
	}, []() {return 0.5 - (rand()/(double) RAND_MAX);});
	
	auto crit = [] __device__ (zhetapi::Vector <double> actual,
			zhetapi::Vector <double> expected) {
		return actual == expected;
	};

	ml::Optimizer <double> *opt = new zhetapi::ml::MeanSquaredError <double> ();

	model.randomize();

	model.set_cost(opt);
	
	ml::opt = 0;
	model.cuda_epochs(ins, outs, 1, 100, 0.1, crit, true);
	ml::opt = 1;
	model.cuda_epochs(ins, outs, 1, 100, 0.1, crit, true);
	model.epochs(ins, outs, 1, 100, 0.1, true);

#endif

}
