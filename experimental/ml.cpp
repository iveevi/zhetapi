#include <iostream>

// Engine headers
#include <dnn.hpp>
#include <training.hpp>

#include <std/activations.hpp>
#include <std/optimizers.hpp>
#include <std/erfs.hpp>
#include <std/linalg.hpp>

using namespace std;
using namespace zhetapi;
using namespace zhetapi::ml;
using namespace zhetapi::utility;

const size_t rounds = 1000000;
const size_t channels = 1;
const size_t actions = 1;

// TODO: add a global variable
Interval <1> unit = 1_I;

DNN <double> decryptor(channels, {
	Layer <double> (4, new ReLU <double> ()),
	Layer <double> (4, new ReLU <double> ()),
	Layer <double> (4, new ReLU <double> ()),
	Layer <double> (4, new ReLU <double> ()),
	Layer <double> (actions, new Sigmoid <double> ())
});

// True signal
auto F = [](Vector <double> sig) {
	return shur(sig, sig);
};

// Communication experiment
int main()
{
	// Optimizer and erf
	Optimizer <double> *opt = new SGD <double> (1);
	Erf <double> *mse = new MSE <double> ();

	// Process
	for (size_t i = 0; i < rounds; i++) {
		Vector <double> sig(actions,
			[&](size_t i) {
				return unit.uniform();
			}
		);
	
		cout << "sig = " << sig << endl;
		cout << "encrypted sig = " << F(sig) << endl;

		Vector <double> in {F(sig)};

		Vector <double> out = decryptor(in);

		cout << "decrytped as " << out << endl;

		cout << "\terror = " << mse->compute(sig, out) << endl;

		fit(decryptor, in, sig, mse, opt);

		// Graph this
	}
}
