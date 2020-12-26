// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>

// Engine headers
#include <std/activation_classes.hpp>
#include <std/optimizer_classes.hpp>
#include <network.hpp>

#include <dataset.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	srand(clock());

	const int size = 1000;
	
	// vector <Vector <double>> ins;
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

	auto crit = [](zhetapi::Vector <double> actual, zhetapi::Vector <double> expected) {
		return actual == expected;
	};

	ml::Optimizer <double> *opt = new zhetapi::ml::MeanSquaredError <double> ();

	model.randomize();

	model.set_cost(opt);
	model.epochs <8> (ins, outs, 10, 250, 0.1, true);

	// Free resources
	delete opt;
}
