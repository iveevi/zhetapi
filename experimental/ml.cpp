#include <iostream>
#include <iomanip>
#include <random>

#include <dnn.hpp>

#include <std/interval.hpp>

#include <std/activations.hpp>
#include <std/erfs.hpp>
#include <std/optimizers.hpp>

using namespace std;
using namespace zhetapi;

using namespace zhetapi::ml;
using namespace zhetapi::utility;

const size_t ITERS = 1000;
const double GAMMA = 0;

template <class T = double>
using Gaussian = normal_distribution <T>;

// GOAL: Return 1, based on actions 1 and 0
int main()
{
	Interval <> i = 1_I;

	// Generators
	random_device rd {};
	mt19937 gen {rd()};

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
		Layer <> (1, new Linear <double> ()),
	});

	// Accumulated reward
	double accumulated = 0;

	const double V = 0.2;
	for (size_t k = 0; k < ITERS; k++) {
		Vector <double> S = rvec();
		Vector <double> P = model(S);

		double Mu = P[0];

		Gaussian <> g(Mu, V);

		double A = g(gen);
		double R = (A > 0 ? 1 : -1);
		double C = (A - Mu)/(2 * V);

		accumulated += R;
		
		// Output
		cout << boolalpha << S << "\t -> "
			<< setw(8) << Mu << "\t-> "
			<< setw(8) << A << "\t-> "
			<< setw(8) << R << "\t-> "
			<< setw(8) << R * C
			<< endl;


		// Use cached version instead
		Matrix <double> *matptr = model.jacobian(S);

		for (size_t i = 0; i < model.size(); i++)
			matptr[i] *= R * C;

		model.apply_gradient(matptr);

		delete[] matptr;
	}

	cout << "\nAverage Reward: " << accumulated/ITERS << endl;
}
