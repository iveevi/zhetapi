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
	srand(clock());

	Vector <double> in(4, 1);
	Vector <double> out(4, 4);

	ml::NeuralNetwork <double> base (4, {
		ml::Layer <double> (4, new ml::ReLU <double> ()),
		ml::Layer <double> (4, new ml::Sigmoid <double> ()),
		ml::Layer <double> (4, new ml::Linear <double> ())
	});

	ml::Erf <double> *cost = new ml::MeanSquaredError <double> ();

	vector <ml::Optimizer <double> *> optimizers {
		new ml::SGD <double> (1),
		new ml::Momentum <double> (0.9, 0.6),
		new ml::Nesterov <double> (0.9, 0.6)
	};

	cout << "out: " << out << endl;
	for (auto opt : optimizers) {
		cout << "\nopt: " << typeid(*opt).name() << endl;

		ml::NeuralNetwork <double> m = base;

		m.set_cost(cost);
		m.set_optimizer(opt);
		
		cout << "\tmodel-out: " << m(in) << endl;

		for (size_t i = 0; i < 10; i++)
			m.fit(in, out);

		cout << "\tmodel-out: " << m(in) << endl;
	}
}
