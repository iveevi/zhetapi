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
	ml::NeuralNetwork <double> cpy;
	ml::NeuralNetwork <double> ld;

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

	model.save("model.out");
	cpy.load("model.out");

	Vector <double> in(11, [](size_t i) {return rand()/((double) RAND_MAX);});
	Vector <double> out(9, [](size_t i) {return rand()/((double) RAND_MAX);});

	cout << "model = " << model(in) << endl;
	cout << "cpy = " << cpy(in) << endl;

	ld.load_json("samples/mnist/model.json");
}
