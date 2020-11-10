#include <cstdlib>
#include <iostream>
#include <random>

#include <activation.hpp>
#include <optimizer.hpp>

#include <std/activation_classes.hpp>
#include <std/optimizer_classes.hpp>

#include <matrix.hpp>
#include <vector.hpp>
#include <network.hpp>
#include <tensor.hpp>

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
	}, []() {return 0.5 - rand()/(double) RAND_MAX;});

	model.randomize();

	// Initialize globals
	Optimizer <double> *opt = new MeanSquaredError <double> ();

	auto input = Vector <double> {1, 1, 1, 1, 1};
	auto target = Vector <double> {0.65, 0.43, 0.29, 0.25, 0.87};

	size_t rounds = 1000;

	// Perform learning
	for (size_t i = 0; i < rounds; i++) {
		cout << "\nRound #" << i + 1 << endl;

		auto result = model(input);

		cout << "\n\t(1, 1) = " << result << endl;
		cout << "\tError = " << (*opt)(target, result)[0] << endl;
		cout << "\tError = " << 100 * (target - result).norm()/result.norm() << "%" << endl;

		model.learn(input, target, opt);
		
		// model.print();
	}

	// Free resources
	delete opt;
}
