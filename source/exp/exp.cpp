#include <network.hpp>

#include <std/activations.hpp>
#include <std/optimizers.hpp>
#include <std/erfs.hpp>
#include <std/initializers.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	srand(clock());

	Vector <double> i = {1, 1, 1};
	Vector <double> o = {0.5, 0.5, 0.5};

	ml::NeuralNetwork <double> model (3, {
		ml::Layer <double> (3, new ml::Sigmoid <double> ()),
		ml::Layer <double> (3, new ml::ReLU <double> ()),
		ml::Layer <double> (3, new ml::Sigmoid <double> ()),
		ml::Layer <double> (3, new ml::ReLU <double> ())
	});

	ml::NeuralNetwork <double> base;

	vector <ml::Optimizer <double> *> opts = {
		new ml::SGD <double> (),
		new ml::Momentum <double> (),
		new ml::Nesterov <double> (),
		new ml::AdaGrad <double> (),
		new ml::RMSProp <double> (),
		new ml::Adam <double> ()
	};

	model.set_cost(new ml::MeanSquaredError <double> ());

	for (auto opt : opts) {
		cout << "Optimizer: " << typeid(*opt).name() << endl;

		base = model;
		base.set_optimizer(opt);

		cout << "\t" << base(i) << endl;

		for (int k = 0; k < 200; k++)
			base.fit(i, o);
		
		cout << "\t" << base(i) << endl;
	}
}
