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

	ml::NeuralNetwork <double> model (4, {
		ml::Layer <double> (4, new ml::ReLU <double> ()),
		ml::Layer <double> (4, new ml::Linear <double> ()),
		ml::Layer <double> (4, new ml::Sigmoid <double> ())
	});

    ml::Erf <double> *cost = new ml::MeanSquaredError <double> ();
    ml::Optimizer <double> *opt = new ml::DefaultOptimizer <double> (1);

    model.set_cost(cost);
    model.set_optimizer(opt);

	Vector <double> in(4, 1);
    Vector <double> out(4, 4);

    cout << "out: " << out << endl;
	cout << "model-out: " << model(in) << endl;

    for (size_t i = 0; i < 100; i++)
        model.fit(in, out);

    cout << "model-out: " << model(in) << endl;
}
