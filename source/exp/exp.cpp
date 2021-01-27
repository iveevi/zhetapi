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
	Vector <double> o = {1, 1, 1};

	ml::NeuralNetwork <double> model (3, {
		ml::Layer <double> (3, new ml::Sigmoid <double> ()),
		ml::Layer <double> (3, new ml::ReLU <double> ())
	});

	model.set_optimizer(new ml::SGD <double> ());
	model.set_cost(new ml::MeanSquaredError <double> ());

	cout << model(i) << endl;

	for (int k = 0; k < 100; k++) {
		model.fit(i, o);

		cout << model(i) << endl;
	}
}
