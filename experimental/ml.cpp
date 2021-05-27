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

const size_t rounds = 5;
const size_t channels = 2;
const size_t actions = 2;

// TODO: add a global variable
Interval <1> unit = 1_I;

DNN <double> decryptor(channels, {
	Layer <double> (4, new ReLU <double> ()),
	/* Layer <double> (4, new ReLU <double> ()),
	Layer <double> (4, new ReLU <double> ()),
	Layer <double> (4, new ReLU <double> ()), */
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

	Optimizer <double> *opt = new Adam <double> (1);
	Erf <double> *mse = new MSE <double> ();
	Erf <double> *dmse = mse->derivative();

	// Normal
	cout << "NORMAL derivative:" << endl;

	Vector <double> out = decryptor(input);
	Vector <double> delta = dmse->compute(out, target);
	Matrix <double> *Jn = decryptor.jacobian_delta(input, delta);

	for (size_t i = 0; i < decryptor.size(); i++)
		linalg::pretty(cout, Jn[i]) << endl;
	
	// Gradient checking
	cout << "CHECK derivative:" << endl;
	Matrix <double> *Jc = decryptor.jacobian_check(input, target, mse);

	for (size_t i = 0; i < decryptor.size(); i++)
		linalg::pretty(cout, Jc[i]) << endl;
}
