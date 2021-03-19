#include <std/filters.hpp>
#include <std/activations.hpp>

using namespace std;
using namespace zhetapi;

int main()
{
	Tensor <double> tensor({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});

	cout << "tensor: " << tensor << endl;
	cout << "cast: " << tensor.cast_to_vector() << endl;

	// Creates pipes
	ml::Pipe <double> pin {
		new Tensor <double> ()
	};

	ml::Pipe <double> pout {
		new Tensor <double> ()
	};

	// Fitler
	ml::FeedForward <double> ff(4, 4, new ml::ReLU <double> ());

	*pin[0] = Vector <double> {1, 2, 3, 4};

	ff.forward_propogate(pin, pout);

	cout << "pin[0] = " << *pin[0] << endl;
	cout << "pout[0] = " << *pout[0] << endl;
}
