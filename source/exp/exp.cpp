// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>

// Engine headers
#include <dataset.hpp>
#include <network.hpp>

// Engine standard headers
#include <std/activations.hpp>
#include <std/erfs.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	ml::ZhetapiRegisterStandardActivations <double> ();

	auto initializer = []() {
                return 0.5 - (rand()/(double) RAND_MAX);
        };

	ml::NeuralNetwork <double> model;

	model = ml::NeuralNetwork <double> ({
                {11, new zhetapi::ml::Linear <double> ()},
                {10, new zhetapi::ml::Sigmoid <double> ()},
                {10, new zhetapi::ml::ReLU <double> ()},
                {10, new zhetapi::ml::Sigmoid <double> ()},
                {9, new zhetapi::ml::Linear <double> ()}
        }, initializer);

	model.randomize();

	ml::Erf <double> *opt = new ml::MeanSquaredError <double> ();

	model.set_cost(opt);

	DataSet <double> ins;
	DataSet <double> outs;

	for (size_t i = 0; i < 50; i++) {
		ins.push_back(Vector <double> (11,
			[](size_t i) {
				return rand()/((double) RAND_MAX);
			}
		));
		
		outs.push_back(Vector <double> (9,
			[](size_t i) {
				return rand()/((double) RAND_MAX);
			}
		));
	}

	model.epochs(ins, outs, 3, 10, 0.001, Display::epoch | Display::graph);
}
