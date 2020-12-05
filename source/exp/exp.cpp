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
		{1, 6},
		{4, 3},
		{5, 8},
		{3, 1}
	};

	vector <Vector <double>> outs = {
		{61, 54},
		{41, 0},
		{65, 4},
		{187, 13}
	};

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

	model.epochs(10, 1, 0.001, opt, ins, outs, crit, false);

	// Free resources
	delete opt;
}
