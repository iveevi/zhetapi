#include <dnn.hpp>

#include <std/activations.hpp>
#include <std/erfs.hpp>
#include <std/optimizers.hpp>

using namespace std;

using namespace zhetapi;
using namespace zhetapi::ml;

int main()
{
	srand(clock());

	DNN <> model(4, {
		Layer <> (4, new Sigmoid <double> ()),
		Layer <> (4, new ReLU <double> ()),
		Layer <> (2, new Sigmoid <double> ())
	});

	model.set_cost(new MeanSquaredError <double> ());
	model.set_optimizer(new Adam <double> ());

	Vector <double> in = {1, 2, 3, 4};

	Matrix <double> *J;
	Matrix <double> *J1;
	Matrix <double> *J2;
	
	J = model.get_gradient(in);

	cout << "J:" << endl;
	for (size_t i = 0; i < model.size(); i++)
		cout << "J[i] = " << J[i] << endl;
	
	J1 = model.get_gradient(in, {1, 0});
	J2 = model.get_gradient(in, {0, 1});

	cout << "Jp:" << endl;
	for (size_t i = 0; i < model.size(); i++)
		cout << "Jp[i] = " << J1[i] + J2[i] << endl;
}
