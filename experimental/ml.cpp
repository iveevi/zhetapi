#include <iostream>
#include <iomanip>

#include <dnn.hpp>

#include <std/interval.hpp>

#include <std/activations.hpp>
#include <std/erfs.hpp>
#include <std/optimizers.hpp>

using namespace std;
using namespace zhetapi;

using namespace zhetapi::ml;
using namespace zhetapi::utility;

const size_t ITERS = 100;
const double GAMMA = 0;

// GOAL: Return 1, based on actions 1 and 0
int main()
{
	Interval <> i = 1_I;

	auto rvec = [&]() -> Vector <double> {
		return {
			i.uniform(),
			i.uniform()
		};
	};

	srand(clock());
	DNN <> model(2, {
		Layer <> (6, new ReLU <double> ()),
		Layer <> (6, new ReLU <double> ()),
		Layer <> (1, new Sigmoid <double> ()),
	});

	for (size_t k = 0; k < ITERS; k++) {
		Vector <double> S = rvec();
		Vector <double> P = model(S);

		bool A = (i.uniform() <= P[0]) ? 1 : 0;

		double R = (A ? 1 : -1);

		// Output
		cout << boolalpha << S << "\t -> "
			<< setw(8) << P[0] << "\t-> "
			<< setw(8) << R
			<< endl;

		// Use cached version instead
		Matrix <double> *matptr = model.get_gradient(S);

		for (size_t i = 0; i < model.size(); i++)
			matptr[i] *= R;

		model.apply_gradient(matptr);

		delete[] matptr;
	}
}
