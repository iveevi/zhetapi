#include <cstdlib>
#include <iostream>
#include <random>

#include "../../engine/activation.hpp"
#include "../../engine/optimizer.hpp"

#include "../../engine/std/activation_classes.hpp"
#include "../../engine/std/optimizer_classes.hpp"

#include "../../engine/matrix.hpp"
#include "../../engine/vector.hpp"
#include "../../engine/network.hpp"
#include "../../engine/tensor.hpp"

using namespace std;

using namespace zhetapi;
using namespace zhetapi::ml;

int main()
{
	// Randomize seed
	srand(clock());

	// Initialize the model
	DeepNeuralNetwork <double> model({
		{5, new Sigmoid <double> ()},
		{5, new ReLU <double> ()}
	}, []() {return 0.5 - (rand()/(double) RAND_MAX);});

	model.randomize();

	// Initialize globals
	Optimizer <double> *opt = new MeanSquaredError <double> ();

	auto input = Vector <double> {1, 1, 1, 1, 1};
	auto target = Vector <double> {0.65, 0.43, 0.29, 0.25, 0.87};

	size_t rounds = 100;

	bool init = true;

	double err_i = 0;
	double err_f = 0;

	Vector <double> out_i {1, 1};
	Vector <double> out_f {1, 1};

	// Perform learning
	for (size_t i = 0; i < rounds; i++) {
		cout << "\nRound #" << i + 1 << endl;

		auto result = model(input);

		cout << "\n\t(1, 1) = " << result << endl;
		cout << "\tError = " << (*opt)(target, result)[0] << endl;

		err_f = 100 * (target - result).norm()/result.norm();
		out_f = result;

		if (init) {
			err_i = err_f;
			out_i = result;

			init = false;
		}

		cout << "\tError = " << err_f << "%" << endl;

		model.learn(input, target, opt, 0.01);
	}

	cout << endl << "================================" << endl;
	cout << "Summary:" << endl;
	cout << "\tInitial:" << endl;
	cout << "\t\tValue: " << out_i << endl;
	cout << "\t\tError: " << err_i << "%" << endl;
	cout << "\tFinal: " << endl;
	cout << "\t\tValue: " << out_f << endl;
	cout << "\t\tError: " << err_f << "%" << endl;

	// Free resources
	delete opt;
}
