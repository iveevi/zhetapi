#include <std/filters.hpp>
#include <std/activations.hpp>
#include <std/erfs.hpp>

using namespace std;
using namespace zhetapi;
using namespace zhetapi::ml;

int main()
{
	/* Tensor <double> tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});

	cout << "tensor: " << tensor << endl;
	cout << "cast: " << tensor.cast_to_vector() << endl; */

	// Creates pipes
	ml::Pipe <double> pin {
		new Tensor <double> ()
	};

	ml::Pipe <double> pout {
		new Tensor <double> ()
	};
	
	ml::Pipe <double> delins {
		new Tensor <double> ()
	};
	
	ml::Pipe <double> grads {
		new Tensor <double> ()
	};

	// Target and cost
	Vector <double> target {5, 6, 7, 8};

	Erf <double> *erf = new SE <double> ();

	// Fitler
	ml::FeedForward <double> ff(4, 4, new ml::Linear <double> ());

	*pin[0] = Vector <double> {1, 2, 3, 4};
	*delins[0] = Vector <double> {1, 1, 1, 1};

	// Iterations
	cout << "LEARNING PROCESS:" << endl;
	for (size_t i = 0; i < 100; i++) {
		cout << string(50, '=') << endl;

		ff.propogate(pin, pout);

		cout << "pin[0] = " << *pin[0] << endl;
		cout << "pout[0] = " << *pout[0] << endl;

		cout << "err: " << erf->compute(pout[0]->cast_to_vector(), target) << endl;

		ff.gradient(delins, grads);

		cout << "delins[0] = " << *delins[0] << endl;
		
		*grads[0] *= 0.001;
		
		cout << "grads[0] = " << *grads[0] << endl;

		ff.apply_gradient(grads);
	}
}
