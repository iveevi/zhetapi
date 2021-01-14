// C/C++ headers
#include <ios>
#include <iostream>
#include <vector>
#include <iomanip>

// Engine headers
#include <network.hpp>

#include <std/activations.hpp>
#include <std/erfs.hpp>
#include <std/optimizers.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	// srand(clock());

	Vector <double> in(2, 1);
	Vector <double> out(4, 4);

	ml::NeuralNetwork <double> base (2, {
		ml::Layer <double> (4, new ml::ReLU <double> ()),
		ml::Layer <double> (4, new ml::Sigmoid <double> ()),
		ml::Layer <double> (4, new ml::Linear <double> ())
	});

	base.set_cost(new ml::MeanSquaredError <double> ());
	base.set_optimizer(new ml::Adam <double> ());

	base.diagnose();
	cout << "base = " << base(in) << endl;
	for (int i = 0; i < 100; i++)
		base.fit(in, out);
	cout << "base = " << base(in) << endl;
	base.diagnose();
}
