// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>
#include <iomanip>

// Engine headers
#include <network.hpp>

#include <std/activations.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	ml::NeuralNetwork <double> model (4, {
		ml::Layer <double> (4, new ml::ReLU <double> ()),
		ml::Layer <double> (4, new ml::Linear <double> ()),
		ml::Layer <double> (4, new ml::Sigmoid <double> ())
	});

	// model.randomize();

	cout << "model-out: " << model({1, 1, 1, 1}) << endl;
}
