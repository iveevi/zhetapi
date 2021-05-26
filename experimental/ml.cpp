#include <iostream>

// Engine headers
#include <dnn.hpp>
#include <std/activations.hpp>

using namespace std;
using namespace zhetapi;
using namespace zhetapi::ml;
using namespace zhetapi::utility;

const size_t rounds = 100;
const size_t channels = 1;
const size_t actions = 1;

// TODO: add a global variable
Interval <1> unit = 1_I;

DNN <double> decryptor(channels, {
	Layer <double> (4, new ReLU <double> ()),
	Layer <double> (4, new ReLU <double> ()),
	Layer <double> (4, new ReLU <double> ()),
	Layer <double> (4, new ReLU <double> ()),
	Layer <double> (actions, new Sigmoid <double> ()),
});

// Communication experiment
int main()
{
	// Initialize random-ness
	srand(clock());

	for (size_t i = 0; i < rounds; i++) {
		cout << "Generated sig: " << unit.uniform() << endl;
	}
}
