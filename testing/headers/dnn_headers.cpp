#include <zhetapi/dnn.hpp>
#include <zhetapi/std/activations.hpp>

using namespace std;
using namespace zhetapi;
using namespace zhetapi::ml;

int main()
{
        DNN <double> model(4, {
		Layer <double> (4, new ReLU <double> ()),
		Layer <double> (4, new ReLU <double> ()),
		Layer <double> (4, new ReLU <double> ())
	});

	cout << "model() = " << model({1, 1, 1, 1}) << endl;
}
