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

int main()
{
	srand(clock());

	const int size = 10;
	
	// vector <Vector <double>> ins;
	DataSet <double> ins;
	DataSet <double> outs;
	
	for (int i = 0; i < size; i++) {
		ins.push_back({(i + 1)/23.0, (i - 0.5)/23.0});

		// printf("ins[%d] = {%f, %f}\n", i, ins[i][0], ins[i][1]);
		
		outs.push_back({rand()/(RAND_MAX * 0.1), rand()/(RAND_MAX * 0.1)});
		
		// printf("outs[%d] = {%f, %f}\n", i, outs[i][0], outs[i][1]);
	}

	ml::NeuralNetwork <double> model ({
		{2, new zhetapi::ml::Linear <double> ()},
		{5, new zhetapi::ml::Linear <double> ()},
		{5, new zhetapi::ml::Linear <double> ()},
		{2, new zhetapi::ml::Linear <double> ()}
	}, []() {return 0.5 - (rand()/(double) RAND_MAX);});

	auto crit = [] __device__ (zhetapi::Vector <double> actual,
			zhetapi::Vector <double> expected) {
		return actual == expected;
	};

	ml::Optimizer <double> *opt = new zhetapi::ml::MeanSquaredError <double> ();

	model.randomize();

	model.set_cost(opt);

	cout << "GPU Training..." << endl;

	model.cuda_epochs <decltype(crit), 1, 2> (ins, outs, 1, 10, 0.1, crit, true);

	cout << endl << "CPU Training..." << endl;
	model.epochs(ins, outs, 1, 10, 0.1, false);

	// cout << "CPU: " << model({1.76, 1.43}) << endl;

	// Free resources
	delete opt;
}
