#include <network.hpp>

#include <std/activations.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	ml::ZhetapiInit <double> ();

	ml::NeuralNetwork model(4, {
		ml::Layer(4, new ml::ReLU <double> ()),
		ml::Layer(4, new ml::ReLU <double> ()),
		ml::Layer(4, new ml::ReLU <double> ())
	});

	model.save("model.out");

	model.print();

	cout << string(50, '=') << endl;

	ml::NeuralNetwork copy;

	copy.load("model.out");

	copy.print();
}
