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
	auto initializer = []() {
                return 0.5 - (rand()/(double) RAND_MAX);
        };

	ml::NeuralNetwork <double> model;
	
	model = ml::NeuralNetwork <double> ({
                {8, new zhetapi::ml::Linear <double> ()},
                {10, new zhetapi::ml::Sigmoid <double> ()},
                {10, new zhetapi::ml::ReLU <double> ()},
                {9, new zhetapi::ml::Linear <double> ()}
        }, initializer);

	model.randomize();

	Vector <double> in(8, [](size_t i) {return rand()/((double) RAND_MAX);});
	Vector <double> out(9, [](size_t i) {return rand()/((double) RAND_MAX);});

	ml::Optimizer <double> *opt = new ml::MeanSquaredError <double> ();

	model.set_cost(opt);

	cout << "in = " << in << endl;
	cout << "out = " << out << endl;

	cout << "model(in) = " << model(in) << endl;

	model.train(in, out, 0.001);
	
	cout << "model(in) = " << model(in) << endl;

	ml::Activation <double> *bolta = new ml::Softmax <double> ();
	ml::Activation <double> *boltb = new ml::SoftmaxInterval <double> ();

	cout << "plain bolta: " << (*bolta)(model(in)) << endl;
	cout << "plain bolta: " << (*boltb)(model(in)) << endl;
	
	cout << "mxb: " << (*bolta)(model(in)).max() << endl;
	cout << "mxa: " << (*boltb)(model(in)).max() << endl;

	delete bolta;
	delete boltb;
}
