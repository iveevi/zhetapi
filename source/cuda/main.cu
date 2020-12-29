// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>

// Engine macros
#define ZHP_GRAD_DEBUG

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

	const int size = 100;
	const int mlen = 1000;
	const int iters = 10;

	ofstream data("data/gpu-cpu-times.dat");

	for (size_t len = 100; len <= mlen; len += 100) {
		cout << "Starting len = " << len << endl;

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

		ts1 = model.cuda_epochs(ins, outs, iters, max(len/10, 1L), 0.1, crit, false);
		ts2 = model.epochs(ins, outs, iters, max(len/10, 1L), 0.1, false);
		ts3 = model.epochs <8> (ins, outs, iters, max(len/10, 1L), 0.1, false);

		data << len
			<< "\t" << ts1.__kernel_time
			<< "\t" << ts2.__kernel_time
			<< "\t" << ts3.__kernel_time << endl;

		// Free resources
		delete opt;
	}

#else

	const int len = 3;
	const int size = 3;
	
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
		{len, new zhetapi::ml::Linear <double> ()},
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
	
	model.cuda_epochs(ins, outs, 2, 100, 0.1, crit, true);
	// model.print();
	model.epochs(ins, outs, 2, 100, 0.1, true);
	// model.print();

#endif

}
