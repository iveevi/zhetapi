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

const size_t rounds = 10;
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

DNN <double> dummy;

// Communication experiment
int main()
{
	// Initialize random-ness
	// srand(clock());

	/* for (size_t i = 0; i < rounds; i++) {
		cout << "Generated sig: " << unit.uniform() << endl;
	} */

	Vector <double> target(actions,
		[&](size_t i) {
			return unit.uniform();
		}
	);
	
	Vector <double> input(channels,
		[&](size_t i) {
			return unit.uniform();
		}
	);

	cout << "target = " << target << endl;
	cout << "input = " << input << endl;

	Optimizer <double> *adam = new Adam <double> ();
	Erf <double> *mse = new MSE <double> ();
	Erf <double> *dmse = mse->derivative();

	// Normal
	dummy = decryptor;

	cout << string(50, '=') << endl;

	Matrix <double> *J = decryptor.jacobian(input);

	// TODO: make pretty printing a string return also
	for (size_t i = 0; i < decryptor.size(); i++) {
		cout << "J[i]:" << endl;
		linalg::pretty(cout, J[i]) << endl;
	}

	delete[] J;

	cout << string(50, '-') << endl;
	cout << "Running " << rounds << " rounds on current jacobians..." << endl;
	for (size_t i = 0; i < rounds; i++) {
		Vector <double> actual = dummy(input);

		cout << "Actual = " << actual << endl;

		Vector <double> delta = dmse->compute(actual, target);
		
		cout << "Delta = " << delta << endl;

		Matrix <double> *Jn = dummy.jacobian_delta(input, delta);

		dummy.apply_gradient(Jn);

		delete[] Jn;
	}

	// Fit training
	dummy = decryptor;
	
	cout << string(50, '-') << endl;
	cout << "Running " << rounds << " rounds on fit method..." << endl;
	for (size_t i = 0; i < rounds; i++) {
		Vector <double> actual = dummy(input);

		cout << "Actual = " << actual << endl;

		Vector <double> delta = dmse->compute(actual, target);
		
		cout << "Delta = " << delta << endl;

		fit(dummy, input, target, mse, adam);
	}
	
	cout << string(50, '=') << endl;
}
