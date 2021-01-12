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
		ml::Layer <double> (4, new ml::Linear <double> ()),
		ml::Layer <double> (4, new ml::Sigmoid <double> ())
	});

    ml::NeuralNetwork <double> m1 = base;
    ml::NeuralNetwork <double> m2 = base;

    ml::Erf <double> *cost = new ml::MeanSquaredError <double> ();
    ml::Optimizer <double> *dopt = new ml::DefaultOptimizer <double> (1);
    ml::Optimizer <double> *mopt = new ml::MomentumOptimizer <double> (0.9, 0.6);

    m1.set_cost(cost);
    m1.set_optimizer(dopt);

    m2.set_cost(cost);
    m2.set_optimizer(mopt);

    cout << "FIRST MODEL:" << endl;

    cout << "out: " << out << endl;
	cout << "model-out: " << m1(in) << endl;

    for (size_t i = 0; i < 100; i++)
        m1.fit(in, out);

    cout << "model-out: " << m1(in) << endl;

    cout << "SECOND MODEL:" << endl;

    cout << "out: " << out << endl;
	cout << "model-out: " << m2(in) << endl;

    for (size_t i = 0; i < 100; i++)
        m2.fit(in, out);

    cout << "model-out: " << m2(in) << endl;
}
