// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>

// Engine headers
#include <std/activation_classes.hpp>
#include <std/optimizer_classes.hpp>
#include <network.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	vector <Vector <double>> ins = {
		{1, 5, 4, 2},
		{1, 6, 4, 2},
		{1, 5, 7, 2},
		{1, 5, 4, 3},
	};

	vector <Vector <double>> outs = {
		{1, 5, 4, 2, 6},
		{1, 6, 4, 3, 1},
		{1, 5, 7, 2, 6},
		{1, 5, 4, 3, 6},
	};

	ml::NeuralNetwork <double> model ({
		{4, new zhetapi::ml::Sigmoid <double> ()},
		{5, new zhetapi::ml::ReLU <double> ()}
	}, []() {return 0.5 - (rand()/(double) RAND_MAX);});

	auto crit = [](zhetapi::Vector <double> actual, zhetapi::Vector <double> expected) {
		return actual == expected;
	};

	ml::Optimizer <double> *opt = new zhetapi::ml::MeanSquaredError <double> ();

	model.epochs(5, 1, 0.1, opt, ins, outs, crit, false);

	// Free resources
	delete opt;
}